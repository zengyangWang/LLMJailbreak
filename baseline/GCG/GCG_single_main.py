import gc

import torch

from baseline.GCG import get_nonascii_toks
from baseline.GCG.minimal_gcg.opt_utils import (
    get_filtered_cands,
    get_logits,
    sample_control,
    target_loss,
    token_gradients,
)
from baseline.GCG.minimal_gcg.string_utils import (
    SuffixManager,
    load_conversation_template,
)
from baseline.GCG.utils import batch_generate, check_attack_success
from utils.test_utils import test_prefixes


class Args:
    def __init__(self, args_dict):
        self.batch_size = 64
        self.topk = 256

        for key, value in args_dict.items():
            setattr(self, key, value)

    def __str__(self):
        attributes = []
        for key in self.__dict__:
            attributes.append(f"{key}={getattr(self, key)}")
        return ", ".join(attributes)


def _build_generation_inputs(suffix_manager, candidates, device):
    """Build prompts ending immediately before the assistant response."""
    generation_inputs = []
    for candidate in candidates:
        input_ids = suffix_manager.get_input_ids(
            adv_string=candidate["adv_suffix"]
        )
        assistant_stop = suffix_manager._assistant_role_slice.stop
        generation_inputs.append(input_ids[:assistant_stop].to(device))
    return generation_inputs


def _print_stage_result(candidate, stage, response, passed):
    print(f"\n{'=' * 80}")
    print(f"Iteration: {candidate['iteration']}")
    print(f"Stage: {stage}")
    print(f"Adversarial suffix: {candidate['adv_suffix']}")
    print(f"Candidate loss: {candidate['loss']}")
    print(f"Model response: {response}")
    print(f"Response length: {len(response)}")
    print(f"Passed: {passed}")
    print(f"{'=' * 80}")


def GCG(
    model,
    tokenizer,
    goal,
    device,
    target,
    args_dict,
    language=None,
):
    """Run GCG and evaluate optimized suffixes in batched two-stage checks.

    One best suffix is retained from each optimization iteration. After
    ``gcg_generate_batch_size`` iterations, all retained suffixes are generated
    together for the short prefilter. Candidates passing the complete short
    response check are then generated together for the full response check.
    """
    args = Args(args_dict)
    adv_string_init = ("! " * args.gcg_suffix).strip()
    num_steps = int(args.gcg_attack_budget)
    optimization_batch_size = int(args.batch_size)
    topk = int(args.topk)
    generation_batch_size = max(
        1, int(getattr(args, "gcg_generate_batch_size", 4))
    )
    prefilter_max_new_tokens = int(
        getattr(args, "gcg_prefilter_max_new_tokens", 32)
    )
    full_max_new_tokens = int(args.target_max_n_tokens)

    if num_steps < 0:
        raise ValueError("gcg_attack_budget must be non-negative")
    if prefilter_max_new_tokens <= 0:
        raise ValueError("gcg_prefilter_max_new_tokens must be greater than zero")
    if full_max_new_tokens <= 0:
        raise ValueError("target_max_n_tokens must be greater than zero")

    allow_non_ascii = False
    template_name = args.template_name
    not_allowed_tokens = None if allow_non_ascii else get_nonascii_toks(tokenizer)
    adv_suffix = adv_string_init
    conv_template = load_conversation_template(template_name, language)
    suffix_manager = SuffixManager(
        tokenizer=tokenizer,
        conv_template=conv_template,
        instruction=goal,
        target=target,
        adv_string=adv_string_init,
    )

    pending_candidates = []
    completion = ""
    is_success = False
    completed_iterations = 0
    successful_iteration = None

    for iteration_index in range(num_steps):
        iteration = iteration_index + 1
        completed_iterations = iteration

        input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix).to(device)
        coordinate_grad = token_gradients(
            model,
            input_ids,
            suffix_manager._control_slice,
            suffix_manager._target_slice,
            suffix_manager._loss_slice,
        )

        current_loss_value = None
        with torch.no_grad():
            control_slice_length = (
                suffix_manager._control_slice.stop
                - suffix_manager._control_slice.start
            )

            if control_slice_length > 0:
                adv_suffix_tokens = input_ids[suffix_manager._control_slice].to(device)
                new_adv_suffix_toks = sample_control(
                    adv_suffix_tokens,
                    coordinate_grad,
                    optimization_batch_size,
                    topk=topk,
                    temp=1,
                    not_allowed_tokens=not_allowed_tokens,
                )
                new_adv_suffix = get_filtered_cands(
                    tokenizer,
                    new_adv_suffix_toks,
                    filter_cand=True,
                    curr_control=adv_suffix,
                )

                logits, candidate_ids = get_logits(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    control_slice=suffix_manager._control_slice,
                    test_controls=new_adv_suffix,
                    return_ids=True,
                )
                losses = target_loss(
                    logits, candidate_ids, suffix_manager._target_slice
                )
                best_candidate_index = int(losses.argmin().item())
                adv_suffix = new_adv_suffix[best_candidate_index]
                current_loss_value = float(
                    losses[best_candidate_index].detach().cpu().item()
                )
            else:
                adv_suffix = adv_string_init

        # The optimization tensors are no longer needed after the best suffix
        # has been selected. Release them before any batched generation so the
        # large candidate logits tensor does not coexist with generation
        # buffers in GPU memory.
        del coordinate_grad
        del input_ids
        if "adv_suffix_tokens" in locals():
            del adv_suffix_tokens
        if "new_adv_suffix_toks" in locals():
            del new_adv_suffix_toks
        if "logits" in locals():
            del logits
        if "candidate_ids" in locals():
            del candidate_ids
        if "losses" in locals():
            del losses
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pending_candidates.append(
            {
                "iteration": iteration,
                "adv_suffix": str(adv_suffix),
                "loss": current_loss_value,
            }
        )

        should_evaluate = (
            len(pending_candidates) >= generation_batch_size
            or iteration == num_steps
        )

        if should_evaluate:
            # Clear Python references and the allocator cache immediately
            # before generation at the end of each group.


            short_inputs = _build_generation_inputs(
                suffix_manager, pending_candidates, device
            )
            short_outputs = batch_generate(
                model,
                tokenizer,
                short_inputs,
                max_new_tokens=prefilter_max_new_tokens,
            )

            passed_candidates = []
            passed_inputs = []
            for candidate_index, (candidate, short_output) in enumerate(
                zip(pending_candidates, short_outputs)
            ):
                short_passed = check_attack_success(
                    short_output, test_prefixes
                )
                _print_stage_result(
                    candidate, "prefilter", short_output, short_passed
                )
                if short_passed:
                    passed_candidates.append(candidate)
                    # The prompt input is identical in the 32-token and
                    # 150-token stages. Reuse the already-tokenized row.
                    passed_inputs.append(short_inputs[candidate_index])

            if short_outputs:
                adv_suffix = pending_candidates[-1]["adv_suffix"]
                completion = short_outputs[-1]

            if passed_candidates:
                full_outputs = batch_generate(
                    model,
                    tokenizer,
                    passed_inputs,
                    max_new_tokens=full_max_new_tokens,
                )

                if full_outputs:
                    adv_suffix = passed_candidates[-1]["adv_suffix"]
                    completion = full_outputs[-1]

                for candidate, full_output in zip(
                    passed_candidates, full_outputs
                ):
                    full_passed = check_attack_success(
                        full_output, test_prefixes
                    )
                    _print_stage_result(
                        candidate, "full", full_output, full_passed
                    )
                    if full_passed:
                        adv_suffix = candidate["adv_suffix"]
                        completion = full_output
                        is_success = True
                        successful_iteration = candidate["iteration"]
                        break

            pending_candidates.clear()
            del short_inputs
            del passed_inputs

        if is_success:
            break

    separator = " " if goal and adv_suffix else ""
    adv_prompt = f"{goal}{separator}{adv_suffix}"
    iterations_used = (
        successful_iteration
        if successful_iteration is not None
        else completed_iterations
    )
    return adv_prompt, completion, iterations_used, is_success
