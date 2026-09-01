import torch
from baseline.GCG import get_nonascii_toks
from baseline.GCG.minimal_gcg.opt_utils import (
    token_gradients,
    sample_control,
    get_logits,
    target_loss,
)
from baseline.GCG.minimal_gcg.opt_utils import get_filtered_cands
from baseline.GCG.minimal_gcg.string_utils import (
    SuffixManager,
    load_conversation_template,
)
import gc
import time
from baseline.GCG.utils import prefix_check_for_attack_success
from utils.test_utils import test_prefixes


class Args:
    def __init__(self, args_dict):
        # self.adv_string_init = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
        # self.num_steps = 500
        self.batch_size = 64
        self.topk = 256

        for k, v in args_dict.items():
            setattr(self, k, v)

    def __str__(self):
        attributes = []
        for k, v in self.__dict__.items():
            attributes.append(f"{k}={getattr(self, k)}")
        return ", ".join(attributes)


# gradicident-based method
def GCG(
    model,
    tokenizer,
    goal,
    device,
    target,
    args_dict,
    language=None,
):
    # initialize
    args = Args(args_dict)
    adv_string_init = "! " * args.gcg_suffix
    adv_string_init = adv_string_init.strip()
    num_steps = args.gcg_attack_budget
    batch_size = args.batch_size
    topk = args.topk

    allow_non_ascii = False  # you can set this to True to use unicode tokens
    template_name = args.template_name
    not_allowed_tokens = None if allow_non_ascii else get_nonascii_toks(tokenizer)
    adv_suffix = adv_string_init
    conv_template = load_conversation_template(template_name, language)
    user_prompt = goal
    target = target

    suffix_manager = SuffixManager(
        tokenizer=tokenizer,
        conv_template=conv_template,
        instruction=user_prompt,
        target=target,
        adv_string=adv_string_init,
    )

    # [METRIC] baseline instrumentation: reset peak-memory stats and start timer
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _t_start = time.time()
    _ph={'grad':0,'logits':0,'gen':0}
    _pb=torch.cuda.memory_allocated()

    # defaults so the return value stays valid even if num_steps == 0
    i = -1
    is_success = False
    completion = ""

    for i in range(num_steps):
        # Step 1. Encode user prompt (behavior + adv suffix) as tokens and return token ids.
        input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix)
        input_ids = input_ids.to(device)

        # Step 2. Compute Coordinate Gradient
        torch.cuda.reset_peak_memory_stats()
        coordinate_grad = token_gradients(
            model,
            input_ids,
            suffix_manager._control_slice,
            suffix_manager._target_slice,
            suffix_manager._loss_slice,
        )
        _ph['grad']=max(_ph['grad'],torch.cuda.max_memory_allocated())

        gen_config = model.generation_config
        gen_config.max_new_tokens = args.target_max_n_tokens

        # Step 3. Sample a batch of new tokens based on the coordinate gradient.
        # Notice that we only need the one that minimizes the loss.
        with torch.no_grad():
            control_slice_length = (
                suffix_manager._control_slice.stop - suffix_manager._control_slice.start
            )
            if control_slice_length > 0:

                # Step 3.1 Slice the input to locate the adversarial suffix.
                adv_suffix_tokens = input_ids[suffix_manager._control_slice].to(device)

                # Step 3.2 Randomly sample a batch of replacements.
                new_adv_suffix_toks = sample_control(
                    adv_suffix_tokens,
                    coordinate_grad,
                    batch_size,
                    topk=topk,
                    temp=1,
                    not_allowed_tokens=not_allowed_tokens,
                )

                # Step 3.3 This step ensures all adversarial candidates have the same number of tokens.
                # This step is necessary because tokenizers are not invertible
                # so Encode(Decode(tokens)) may produce a different tokenization.
                # We ensure the number of token remains to prevent the memory keeps growing and run into OOM.
                new_adv_suffix = get_filtered_cands(
                    tokenizer,
                    new_adv_suffix_toks,
                    filter_cand=True,
                    curr_control=adv_suffix,
                )
                input_ids_temp = suffix_manager.get_input_ids(adv_string=adv_suffix).to(
                    device
                )
                # Step 3.4 Compute loss on these candidates and take the argmin.
                torch.cuda.reset_peak_memory_stats()
                logits, ids = get_logits(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    control_slice=suffix_manager._control_slice,
                    test_controls=new_adv_suffix,
                    return_ids=True,
                )

                losses = target_loss(logits, ids, suffix_manager._target_slice)
                _ph['logits']=max(_ph['logits'],torch.cuda.max_memory_allocated())

                best_new_adv_suffix_id = losses.argmin()
                best_new_adv_suffix = new_adv_suffix[best_new_adv_suffix_id]

                current_loss = losses[best_new_adv_suffix_id]

                # Update the running adv_suffix with the best candidate
                adv_suffix = best_new_adv_suffix
            else:
                adv_suffix = adv_string_init

            torch.cuda.reset_peak_memory_stats()
            is_success, gen_str = prefix_check_for_attack_success(
                model,
                tokenizer,
                suffix_manager.get_input_ids(adv_string=adv_suffix).to(device),
                suffix_manager._assistant_role_slice,
                test_prefixes,
                gen_config=gen_config,
            )
            _ph['gen']=max(_ph['gen'],torch.cuda.max_memory_allocated())

        completion = gen_str

        print(f"""\n{'=' * 80}\n  Iteration: {i}\n{'=' * 80}\n""")
        print(f"\n User Query: {goal}")
        print(f"\n Adversarial Prompt: {adv_suffix}")
        print(f"\n Model Response: {completion}")
        print(print(f"\n Length of Model Response: {len(completion)}"))
        print(f"\n Is Success: {is_success}")
        print(f"""\n{'=' * 80}\n Iteration: {i}\n{'=' * 80}\n""")
        # Notice that for the purpose of demo we stop immediately if we pass the checker but you are free to
        # comment this to keep the optimization running for longer (to get a lower loss).
        if is_success:
            break

        # (Optional) Clean up the cache.
        # del coordinate_grad, adv_suffix_tokens
        if "coordinate_grad" in locals():
            del coordinate_grad

        if "adv_suffix_tokens" in locals():
            del adv_suffix_tokens

        if "logits" in locals():
            del logits
        if "ids" in locals():
            del ids
        if "losses" in locals():
            del losses
        gc.collect()
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()

    # 构造最终字符串并直接返回（不再进行额外验证/推理）
    separator = " " if user_prompt else ""
    full_prompt_text = f"{user_prompt}{separator}{adv_suffix}"
    adv_prompt = full_prompt_text
    # [METRIC] four comparison metrics for this prompt
    _elapsed = time.time() - _t_start
    _peak = (
        max(_ph.values()) / 1024 ** 3
        if torch.cuda.is_available()
        else 0.0
    )
    print(
        f"[METRIC] peak_mem={_peak:.3f}GiB elapsed={_elapsed:.1f}s "
        f"iterations={i + 1} success={is_success} "
        f"batch_size={batch_size} topk={topk} num_steps={num_steps}",
        flush=True,
    )
    G=1024**3
    print('[PHASE] base=%.3fGiB grad=%.3fGiB logits=%.3fGiB gen=%.3fGiB'%(_pb/G,_ph['grad']/G,_ph['logits']/G,_ph['gen']/G),flush=True)

    return adv_prompt, completion, i + 1, is_success
