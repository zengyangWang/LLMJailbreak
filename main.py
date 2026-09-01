import os
import argparse
import sys
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
BUNDLE_ROOT = os.path.dirname(REPO_ROOT)
os.environ.setdefault("NLTK_DATA", os.path.join(BUNDLE_ROOT, "nltk_data"))
VENDORED_FASTCHAT_ROOT = os.path.join(REPO_ROOT, "FastChat")
if VENDORED_FASTCHAT_ROOT not in sys.path:
    sys.path.insert(0, VENDORED_FASTCHAT_ROOT)

# 设置 vLLM 多进程启动方式为 spawn，以避免在已初始化 CUDA 的进程中 fork 导致的 RuntimeError
os.environ["VLLM_WORKER_MULTIPROCESSING_START_METHOD"] = "spawn"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from typing import List
from tqdm import tqdm
import datetime
from data.language import Language

from utils.utils import load_model_and_tokenizer, set_random_seed
from utils.string_utils import load_prompts, load_goals
from utils.test_utils import (
    get_template_name,
    save_test_to_file,
    load_test_from_file,
    load_test_from_file_split,
    save_test_to_file_split,
    load_split_file_whole,
    instruction2dratk_data_path,
    export_results_table,
)

# 导入参数初始化模块
from initialize_args import initialize_args

# 导入各种攻击基线方法 (Attack Baselines)
from baseline.GCG.GCG_single_main import GCG
from baseline.AutoDAN.AutoDAN_single_main import AutoDAN_single_main
from baseline.TAP.TAP_single_main import TAP_single_main, TAP_initial
from baseline.PAIR.PAIR_single_main import PAIR_single_main, PAIR_initial
from baseline.GPTFuzz.GPTFuzz_single_main import GPTFuzz_initial, GPTFuzz_single_main
from baseline.AmpleGCG.AmpleGCG_single_main import (
    AmpleGCG_initial,
    AmpleGCG_single_main,
    AmpleGCG_generate_suffix,
)
from baseline.AdvPrompter.AdvPrompter_single_main import (
    AdvPrompter_initial,
    AdvPrompter_generate_suffix,
    AdvPrompter_single_main,
)
from baseline.Adaptive import LLMAdaptive_initial, LLMAdaptive_single_main
from baseline.AmpleGCG.utils import load_target_models_amplegcg
from baseline.AdvPrompter.utils import load_target_models_advprompter
from baseline.DrAttack.DrAttack_single_main import (
    DrAttack_initial,
    DrAttack_single_main,
    DrAttack_stop,
)
from baseline.MultiJail.MultiJail_single_main import (
    MultiJail_initial,
    MultiJail_single_main,
    MultiJail_generate_suffix,
)
from baseline.MultiJail.utils import load_target_models_MultiJail
from baseline.PAP.PAP_single_main import PAP_initial, PAP_single_main
from baseline.JailBroken import JailBroken_initial, JailBroken_single_main
from baseline.MJP import MJP_initial, MJP_single_main
from baseline.Coldattack.Coldattack_single_main import (
    Coldattack_initial,
    Coldattack_single_main,
)
from baseline.ReNeLLM.ReNeLLM_single_main import ReNeLLM_single_main
from baseline.ReNeLLM.utils import load_target_models_renellm
from baseline.Template.Template_single_main import (
    Template_single_main,
)
from baseline.Translate.Translate_single_main import (
    Translate_single_main,
)
from baseline.SensitivePinyin.SensitivePinyin_single_main import (
    SensitivePinyin_single_main,
)
from baseline.NoiseInjection.NoiseInjection_single_main import (
    NoiseInjection_single_main,
)
from baseline.ICA.ICA_single_main import ICA_initial, ICA_single_main
from baseline.Actorattack.Actorattack_single_main import Actorattack_single_main
from baseline.SATA.SATA_single_main import SATA_initial, SATA_single_main
from baseline.Cipher.Cipher_single_main import Cipher_single_main
from baseline.CodeChameleon.CodeChameleon_single_main import (
    CodeChameleon_single_main,
)
from baseline.FlipAttack.FlipAttack_single_main import FlipAttack_single_main
from baseline.FuzzLLM.FuzzLLM_single_main import FuzzLLM_single_main

# 导入防御方法
from defense import test_smoothLLM, generate_defense_goal

# 导入评估代理 (Evaluation Agent)
from GPTEvaluatorAgent.agent_eval import agent_evaluation


def generate_attack_result(
    goal: str,
    target: str,
    language: Language,
    models: List,
    device: str,
    args: dict,
    curr_output: dict,
):
    """
    根据指定的攻击方法执行单次攻击，并返回结果。

    Args:
        goal (str): 攻击目标提示词 (可能经过防御处理)。
        target (str): 期望的攻击成功输出开头。
        models (list): 包含加载的模型、Tokenizer 等。
        device (str): 运行设备 (如 'cuda:0')。
        args: 命令行参数对象。
        curr_output (dict): 当前的数据记录字典，用于存储结果。

    Returns:
        dict: 更新后的 curr_output，包含攻击生成的 prompt、模型输出等。
    """
    # 根据 args.attack 选择对应的攻击逻辑
    if args.attack == "GCG":
        model, tokenizer = models[0], models[1]
        curr_args_dict = vars(args)
        # 执行 GCG 攻击
        adv_prompt, model_output, iteration, is_JB = GCG(
            model=model,
            tokenizer=tokenizer,
            device=device,
            goal=goal,
            target=target,
            args_dict=curr_args_dict,
            language=language,
        )
        curr_output["adv_prompt"] = adv_prompt
        curr_output["language_model_output"] = model_output
        curr_output["attack_iterations"] = iteration
        curr_output["is_JB"] = is_JB

    elif args.attack == "AutoDAN":
        model, tokenizer = models[0], models[1]
        curr_args_dict = vars(args)
        # 执行 AutoDAN 攻击
        adv_prompt, model_output, iteration, is_JB = AutoDAN_single_main(
            args_dict=curr_args_dict,
            target_model=model,
            target_tokenizer=tokenizer,
            goal=goal,
            target=target,
            language=language.name,
        )
        curr_output["adv_prompt"] = adv_prompt
        curr_output["language_model_output"] = model_output
        curr_output["attack_iterations"] = iteration
        curr_output["is_JB"] = is_JB

    elif args.attack == "AmpleGCG":
        model = models[0]
        curr_args_dict = vars(args)
        # 仅生成 Prompt（不执行推理）
        result_dict = AmpleGCG_single_main(
            args_dict=curr_args_dict,
            target_model=model,
            goal=goal,
            target=target,
            language=language,
        )
        all_prompts = result_dict.get("all_prompts", [])
        # 新字段：保存全部生成的 Prompt
        curr_output["all_generated_prompts"] = all_prompts
        # 兼容旧字段
        curr_output["adv_prompt"] = all_prompts[0] if len(all_prompts) > 0 else "NULL"
        curr_output["language_model_output"] = "SKIPPED (Generation Only)"
        curr_output["is_JB"] = False

    elif args.attack == "AdvPrompter":
        model = models[0]
        curr_args_dict = vars(args)
        # 仅生成 Prompt（不执行推理）
        result_dict = AdvPrompter_single_main(
            args_dict=curr_args_dict,
            target_model=model,
            goal=goal,
            target=target,
            language=language,
        )
        all_prompts = result_dict.get("all_prompts", [])
        curr_output["all_generated_prompts"] = all_prompts
        curr_output["adv_prompt"] = all_prompts[0] if len(all_prompts) > 0 else "NULL"
        curr_output["language_model_output"] = "SKIPPED (Generation Only)"
        curr_output["is_JB"] = False

    elif args.attack == "LLMAdaptive":
        target_model = models[0]
        curr_args_dict = vars(args)
        result = LLMAdaptive_single_main(
            args_dict=curr_args_dict,
            target_model=target_model,
            goal=goal,
            target=target,
            language=language,
        )
        curr_output["adv_prompt"] = result.get("adv_prompt", "")
        curr_output["language_model_output"] = result.get("language_model_output", "")
        curr_output["attack_iterations"] = result.get("attack_iterations")
        curr_output["is_JB"] = result.get("is_JB", False)
    elif args.attack == "JailBroken":
        attack_model = models[0]
        curr_args_dict = vars(args)
        result = JailBroken_single_main(
            args_dict=curr_args_dict,
            attack_model=attack_model,
            goal=goal,
            target=target,
            language=language,
        )
        curr_output["adv_prompt"] = result.get("adv_prompt", "")
        curr_output["language_model_output"] = result.get("language_model_output", "")
        curr_output["attack_iterations"] = result.get("attack_iterations")
        curr_output["is_JB"] = result.get("is_JB", False)
    elif args.attack == "MJP":
        adapter, generator, templates = models[0], models[1], models[2]
        curr_args_dict = vars(args)
        result = MJP_single_main(
            args_dict=curr_args_dict,
            adapter=adapter,
            generator=generator,
            templates=templates,
            goal=goal,
            target=target,
            language=language,
        )
        curr_output["adv_prompt"] = result.get("adv_prompt", "")
        curr_output["language_model_output"] = result.get("language_model_output", "")
        curr_output["attack_iterations"] = result.get("attack_iterations")
        curr_output["is_JB"] = result.get("is_JB", False)

    elif args.attack == "DrAttack":
        curr_args_dict = vars(args)
        # 执行 DrAttack 攻击
        curr_output_record = DrAttack_single_main(
            args_dict=curr_args_dict,
            worker=models[0],
            attack=models[1],
            goal=goal,
            language=language,
        )
        # DrAttack 返回的是字典，需要提取字段
        curr_output["adv_prompt"] = curr_output_record["adv_prompt"]
        curr_output["optimized_sentence"] = curr_output_record["optimized_sentence"]
        curr_output["language_model_output"] = curr_output_record[
            "language_model_output"
        ]
        curr_output["negative_similarity_score"] = curr_output_record[
            "negative_similarity_score"
        ]
        curr_output["attack_iterations"] = curr_output_record["attack_iterations"]
        curr_output["is_JB"] = curr_output_record["is_JB"]

    elif args.attack == "MultiJail":
        model = models[0]
        curr_args_dict = vars(args)
        # 执行 MultiJail 攻击
        adv_prompt, model_output, iteration, is_JB = MultiJail_single_main(
            args_dict=curr_args_dict,
            target_model=model,
            goal=goal,
            target=target,
        )
        curr_output["adv_prompt"] = adv_prompt
        curr_output["language_model_output"] = model_output
        curr_output["attack_iterations"] = iteration
        curr_output["is_JB"] = is_JB

    elif args.attack == "TAP":
        attack_llm, target_llm, evaluator_llm = models[0], models[1], models[2]
        curr_args_dict = vars(args)
        # 执行 TAP 攻击
        curr_output_record = TAP_single_main(
            args_dict=curr_args_dict,
            attack_llm=attack_llm,
            target_llm=target_llm,
            evaluator_llm=evaluator_llm,
            goal=goal,
            target=target,
            language=language,
        )
        # TAP 返回包含详细信息的字典
        curr_output["language_model_output"] = curr_output_record[
            "language_model_output"
        ]
        curr_output["attack_iterations"] = curr_output_record["attack_iterations"]
        curr_output["is_JB"] = curr_output_record["is_JB"]
        curr_output["is_JB_Judge"] = curr_output_record["is_JB_Judge"]
        curr_output["attack_prompt"] = curr_output_record["attack_prompt"]
        curr_output["improve_prompt"] = curr_output_record["improve_prompt"]
        curr_output["judge_output"] = curr_output_record["judge_output"]
        curr_output["on_topic_score"] = curr_output_record["on_topic_score"]

    elif args.attack == "PAIR":
        attack_llm, target_llm, evaluator_llm = models[0], models[1], models[2]
        curr_args_dict = vars(args)
        # 执行 PAIR 攻击
        curr_output_record = PAIR_single_main(
            args_dict=curr_args_dict,
            attackLM=attack_llm,
            targetLM=target_llm,
            evaluator=evaluator_llm,
            goal=goal,
            target=target,
            language=language,
        )
        # PAIR 返回包含详细信息的字典
        curr_output["language_model_output"] = curr_output_record[
            "language_model_output"
        ]
        curr_output["attack_iterations"] = curr_output_record["attack_iterations"]
        curr_output["is_JB"] = curr_output_record["is_JB"]
        curr_output["is_JB_Judge"] = curr_output_record["is_JB_Judge"]
        curr_output["attack_prompt"] = curr_output_record["attack_prompt"]
        curr_output["improve_prompt"] = curr_output_record["improve_prompt"]
        curr_output["judge_output"] = curr_output_record["judge_output"]

    elif args.attack == "GPTFuzz":
        openai_model = models[0] if models else None
        curr_args_dict = vars(args)
        # 执行 GPTFuzz 攻击（仅返回改写后的 prompt，不使用目标/判定模型）
        curr_output_record = GPTFuzz_single_main(
            args_dict=curr_args_dict,
            openai_model=openai_model,
            target_model=None,
            predictor_model=None,
            goal=goal,
            target=target,
            language=language.name,
        )
        curr_output["language_model_output"] = curr_output_record[
            "language_model_output"
        ]
        curr_output["attack_iterations"] = curr_output_record["attack_iterations"]
        curr_output["is_JB"] = curr_output_record["is_JB"]
        curr_output["is_JB_Judge"] = curr_output_record["is_JB_Judge"]
        curr_output["attack_prompt"] = curr_output_record["attack_prompt"]

    elif args.attack == "PAP":
        attack_model = models[0]
        curr_args_dict = vars(args)
        # 执行 PAP 攻击
        curr_output_record = PAP_single_main(
            args_dict=curr_args_dict,
            attack_model=attack_model,
            goal=goal,
            target=target,
            language=language,
        )
        curr_output["adv_prompt"] = curr_output_record["adv_prompt"]
    elif args.attack == "ReNeLLM":
        target_model = models[0]
        curr_args_dict = vars(args)
        # 仅生成 Prompt（不执行推理）
        result_dict = ReNeLLM_single_main(
            args_dict=curr_args_dict,
            target_model=target_model,
            goal=goal,
            target=target,
            language=language.name,
        )
        all_prompts = result_dict.get("all_prompts", [])
        # 保存全部生成的 Prompt
        curr_output["all_generated_prompts"] = all_prompts
        # 兼容旧字段
        curr_output["adv_prompt"] = all_prompts[0] if len(all_prompts) > 0 else "NULL"
        curr_output["language_model_output"] = "SKIPPED (Generation Only)"
        curr_output["is_JB"] = False
    elif args.attack == "Cipher":
        curr_args_dict = vars(args)
        curr_output_record = Cipher_single_main(
            args_dict=curr_args_dict,
            goal=goal,
            target=target,
            language=language,
        )
        curr_output["adv_prompt"] = curr_output_record["adv_prompt"]
    elif args.attack == "Coldattack":
        target_model, target_tokenizer = models[0], models[1]
        curr_args_dict = vars(args)
        adv_prompt, model_output, iteration, is_JB = Coldattack_single_main(
            args_dict=curr_args_dict,
            target_model=target_model,
            target_tokenizer=target_tokenizer,
            goal=goal,
            target=target,
            language=language,
        )
        curr_output["adv_prompt"] = adv_prompt
        curr_output["language_model_output"] = model_output
        curr_output["attack_iterations"] = iteration
        curr_output["is_JB"] = is_JB
    elif args.attack == "Template":
        target_model = models[0] if len(models) > 0 else None
        curr_args_dict = vars(args)
        curr_output_record = Template_single_main(
            args_dict=curr_args_dict,
            target_model=target_model,
            goal=goal,
            target=target,
            language=language,
        )
        curr_output["adv_prompt"] = curr_output_record["adv_prompt"]
        curr_output["language_model_output"] = curr_output_record[
            "language_model_output"
        ]
        curr_output["is_JB"] = curr_output_record["is_JB"]
        curr_output["attack_iterations"] = curr_output_record.get("attack_iterations")
        # 数据产出模式：不评估、不记录成功细节
    elif args.attack == "SensitivePinyin":
        curr_args_dict = vars(args)
        curr_output_record = SensitivePinyin_single_main(
            args_dict=curr_args_dict,
            goal=goal,
            target=target,
            language=language,
        )
        all_prompts = curr_output_record.get("all_generated_prompts", []) or []
        curr_output["all_generated_prompts"] = all_prompts
        curr_output["adv_prompt"] = curr_output_record.get("adv_prompt", "NULL")
        curr_output["language_model_output"] = "SKIPPED (Generation Only)"
        curr_output["is_JB"] = False
        curr_output["attack_iterations"] = curr_output_record.get("attack_iterations")
    elif args.attack == "NoiseInjection":
        curr_args_dict = vars(args)
        curr_output_record = NoiseInjection_single_main(
            args_dict=curr_args_dict,
            goal=goal,
            target=target,
            language=language,
        )
        all_prompts = curr_output_record.get("all_generated_prompts", []) or []
        curr_output["all_generated_prompts"] = all_prompts
        curr_output["adv_prompt"] = curr_output_record.get("adv_prompt", "NULL")
        curr_output["language_model_output"] = "SKIPPED (Generation Only)"
        curr_output["is_JB"] = False
        curr_output["attack_iterations"] = curr_output_record.get("attack_iterations")
    elif args.attack == "Translate":
        target_model = models[0] if len(models) > 0 else None
        curr_args_dict = vars(args)
        curr_output_record = Translate_single_main(
            args_dict=curr_args_dict,
            target_model=target_model,
            goal=goal,
            target=target,
            language=language,
        )
        curr_output["adv_prompt"] = curr_output_record["adv_prompt"]
        curr_output["language_model_output"] = curr_output_record[
            "language_model_output"
        ]
        curr_output["is_JB"] = curr_output_record["is_JB"]
        curr_output["attack_iterations"] = curr_output_record.get("attack_iterations")
        # 数据产出模式：不评估、不记录成功细节
    elif args.attack == "ICA":
        # 仅生成 Prompt（不执行推理）
        curr_args_dict = vars(args)
        result = ICA_single_main(
            args_dict=curr_args_dict,
            target_model=models[0] if len(models) > 0 else None,
            goal=goal,
            target=target,
            language=language,
        )
        curr_output["adv_prompt"] = result.get("adv_prompt", "")
        curr_output["language_model_output"] = "SKIPPED (Generation Only)"
        curr_output["is_JB"] = False
    elif args.attack == "SATA":
        # 仅生成 Prompt（不执行目标模型推理）；使用本地 Vicuna 作为攻击模型生成 mask
        attack_model, attack_tokenizer = (
            (models[0], models[1]) if len(models) >= 2 else (None, None)
        )
        curr_args_dict = vars(args)
        curr_output_record = SATA_single_main(
            args_dict=curr_args_dict,
            attack_model=attack_model,
            attack_tokenizer=attack_tokenizer,
            goal=goal,
            target=target,
            language=language,
        )
        curr_output["adv_prompt"] = curr_output_record["adv_prompt"]
        curr_output["language_model_output"] = curr_output_record[
            "language_model_output"
        ]
        curr_output["is_JB"] = curr_output_record["is_JB"]
    elif args.attack == "CodeChameleon":
        curr_args_dict = vars(args)
        curr_output_record = CodeChameleon_single_main(
            args_dict=curr_args_dict,
            goal=goal,
            target=target,
            language=language,
        )
        curr_output["adv_prompt"] = curr_output_record["adv_prompt"]
    elif args.attack == "FlipAttack":
        curr_args_dict = vars(args)
        curr_output_record = FlipAttack_single_main(
            args_dict=curr_args_dict,
            goal=goal,
            target=target,
            language=language,
        )
        curr_output["adv_prompt"] = curr_output_record["adv_prompt"]
    elif args.attack == "Actorattack":
        curr_args_dict = vars(args)
        # Actorattack handles its own model loading via path in args
        result = Actorattack_single_main(
            args_dict=curr_args_dict,
            target_model=None,
            goal=goal,
            target=target,
            language=language,
        )
        curr_output["adv_prompt"] = result.get("adv_prompt", "")
        curr_output["language_model_output"] = result.get("language_model_output", "")
        curr_output["attack_iterations"] = result.get("attack_iterations")
        curr_output["is_JB"] = result.get("is_JB", False)
    elif args.attack == "FuzzLLM":
        curr_args_dict = vars(args)
        curr_output_record = FuzzLLM_single_main(
            args_dict=curr_args_dict,
            goal=goal,
            target=target,
            language=language,
        )
        curr_output["adv_prompt"] = curr_output_record["adv_prompt"]
    else:
        raise NameError
    return curr_output


def test(
    goals: List[str],
    targets: List[str],
    languages: List[Language],
    models,
    device: str,
    args: dict,
    all_output: List[dict] = [],
):  # 多加language字段，用于中文适配
    """
    执行测试循环，遍历所有目标进行攻击和评估。

    Args:
        goals (list): 所有的攻击目标列表。
        targets (list): 所有的预期输出目标列表。
        models (list): 初始化的模型列表。
        device (str): 设备信息。
        args: 参数对象。
        languages (list): 对应的语言列表。
        all_output (list): 用于存储所有测试结果的列表 (用于断点续传)。

    Returns:
        list: 包含所有测试结果的列表。
    """

    # 1. 生成防御目标 (Perturbed Goals)
    # 如果是 DrAttack 且使用了特定的防御，需要从特定路径加载目标
    if args.attack == "DrAttack" and args.defense_type in [
        "self_reminder",
        "RPO",
        "smoothLLM",
    ]:
        instruction_name = args.instructions_path.split("/")[-1]
        pert_goals_path = instruction2dratk_data_path[instruction_name][
            args.defense_type
        ]
        # pert_goals = load_pert_goals(pert_goals_path)
        pert_goals = load_goals(pert_goals_path)
    else:
        # 否则，根据 defense_type 动态生成防御后的目标 (pert_goals)
        pert_goals = [
            generate_defense_goal(
                goal_i,
                defense_type=args.defense_type,
                pert_type=args.pert_type,
                smoothllm_pert_pct=args.smoothllm_pert_pct,
            )
            for goal_i in goals
        ]

    # 2. 为某些特定攻击方法预先生成后缀 (Suffix)
    if args.attack == "AmpleGCG":
        args.suffix_dict = AmpleGCG_generate_suffix(args, pert_goals)
        models = [load_target_models_amplegcg(args)]
    elif args.attack == "AdvPrompter":
        # 按需求：后缀生成使用原始英文 goal，以保证 AdvPrompter 质量
        args.suffix_dict = AdvPrompter_generate_suffix(args, goals)
        models = [load_target_models_advprompter(args)]
    elif args.attack == "MultiJail":
        # MultiJail 需要先用攻击侧模型为每个样本生成后缀，再由本地 vicuna 目标模型进行测试
        args.suffix_dict = MultiJail_generate_suffix(args, pert_goals)
        models = [load_target_models_MultiJail(args)]

    # 3. 主循环：遍历每个数据点进行测试
    for goal_i, target_i, language_i, pert_goal_i in tqdm(
        zip(
            goals[args.test_data_idx : args.end_index],
            targets[args.test_data_idx : args.end_index],
            languages[args.test_data_idx : args.end_index],
            pert_goals[args.test_data_idx : args.end_index],
        ),
        desc="Testing",
    ):
        # 每一个 goal, target, pert_goal 都是一次攻击的输入
        print(f"""\n{'=' * 36}\nDefense Method: {args.defense_type}\n{'=' * 36}\n""")

        if args.attack == "AdvPrompter":
            # 不进行语言包装，直接透传原始 goal
            goal_to_use_i = goal_i
            # Target Priming: 根据语言设定 prime_text（可选保留）
            lang_name = getattr(language_i, "name", str(language_i)).upper()
            prime_map = {
                "ENGLISH": "Sure, here is",
                "CHINESE_SIMPLIFIED": "好的，这是",
                "CHINESE_TRADITIONAL": "好的，這是",
                "FRENCH": "Bien sûr, voici",
                "GERMAN": "Gern, hier ist",
                "SPANISH": "Claro, aquí está",
                "ITALIAN": "Certo, ecco",
                "JAPANESE": "はい、こちらは",
            }
            args.prime_text = prime_map.get(lang_name, "Sure, here is")
        elif args.attack == "ReNeLLM":
            # 对于 ReNeLLM，不在此处做语言包装，直接沿用原始 goal
            goal_to_use_i = goal_i
        else:
            goal_to_use_i = pert_goal_i

        # 初始化当前数据点的结果记录
        curr_output = {
            "original_prompt": goal_i,
            "perturbed_prompt": goal_to_use_i,
            "target": target_i,
            "language": language_i.name,
            "adv_prompt": "NULL",
            "language_model_output": "NULL",
            "attack_iterations": None,
            "data_id": args.test_data_idx,
            "is_JB": "None",
            "is_JB_Judge": "None",
            "is_JB_Agent": "None",
        }
        print(curr_output)

        print(f"""\n{'=' * 36}\nAttack Method: {args.attack}\n{'=' * 36}\n""")

        # 调用攻击生成函数
        curr_output = generate_attack_result(
            goal_to_use_i, target_i, language_i, models, device, args, curr_output
        )

        print(f"""\n{'=' * 36}\nFinish testing data_id: {args.test_data_idx}\n""")
        print(curr_output)
        print(f"""{'=' * 36}\n""")

        # 保存结果
        all_output.append(curr_output)
        if args.data_split:
            save_test_to_file_split(args=args, instruction=curr_output)
        else:
            save_test_to_file(args=args, instructions=all_output)

        # 增加索引，准备下一个
        args.test_data_idx += 1

    return all_output


def run(
    goals: List[str],
    targets: List[str],
    languages: List[Language],
    target_model_path: str,
    device: str,
    args: dict,
    all_output=[],
):
    """
    初始化模型并启动测试流程。

    Args:
        goals (list): 攻击目标。
        targets (list): 预期输出。
        languages (list): 对应的语言列表。
        target_model_path (str): 目标模型路径。
        device (str): 设备。
        args: 参数。
        all_output (list): 结果列表。
    """

    # 1. 根据攻击方法加载所需的模型 (Target Model, Attack Model, Evaluator Model)
    if args.attack in ["GCG", "AutoDAN"]:
        target_model, target_tokenizer = load_model_and_tokenizer(
            target_model_path, tokenizer_path=None, device=device
        )
        models = [target_model, target_tokenizer]
    elif args.attack == "TAP":
        args_dict = vars(args)
        args, attack_llm, target_llm, evaluator_llm = TAP_initial(args_dict=args_dict)
        models = [attack_llm, target_llm, evaluator_llm]
    elif args.attack == "PAIR":
        args_dict = vars(args)
        args, attack_llm, target_llm, evaluator_llm = PAIR_initial(args_dict=args_dict)
        models = [attack_llm, target_llm, evaluator_llm]
    elif args.attack == "GPTFuzz":
        args_dict = vars(args)
        args, openai_model, _, _ = GPTFuzz_initial(args_dict=args_dict)
        models = [openai_model]
    elif args.attack == "wild_adv_prompt":
        models = []
    elif args.attack == "AmpleGCG":
        args_dict = vars(args)
        args = AmpleGCG_initial(args_dict=args_dict)
        models = []
    elif args.attack == "AdvPrompter":
        args_dict = vars(args)
        args = AdvPrompter_initial(args_dict=args_dict)
        models = []
    elif args.attack == "DrAttack":
        args_dict = vars(args)
        args, worker, attack = DrAttack_initial(args_dict=args_dict)
        models = [worker, attack]
    elif args.attack == "MultiJail":
        args_dict = vars(args)
        args = MultiJail_initial(args_dict=args_dict)
        models = []
    elif args.attack == "PAP":
        args_dict = vars(args)
        attack_model = PAP_initial(args_dict=args_dict)
        models = [attack_model]
    elif args.attack == "LLMAdaptive":
        args_dict = vars(args)
        args, target_model = LLMAdaptive_initial(args_dict=args_dict)
        models = [target_model]
    elif args.attack == "ReNeLLM":
        # 同步进化次数参数到通用字段，便于下游组件读取
        try:
            if getattr(args, "renellm_evo_max", None) is not None:
                setattr(args, "evo_max", int(args.renellm_evo_max))
        except Exception:
            pass
        # 加载 ReNeLLM 目标侧封装模型
        target_llm = load_target_models_renellm(args)
        models = [target_llm]
    elif args.attack == "JailBroken":
        args_dict = vars(args)
        args, attack_model = JailBroken_initial(args_dict=args_dict)
        models = [attack_model]
    elif args.attack == "MJP":
        args_dict = vars(args)
        args, adapter, generator, templates = MJP_initial(args_dict=args_dict)
        models = [adapter, generator, templates]
    elif args.attack == "Coldattack":
        args_dict = vars(args)
        target_model, target_tokenizer = Coldattack_initial(
            args_dict=args_dict, device=device
        )
        models = [target_model, target_tokenizer]
    elif args.attack == "ICA":
        args_dict = vars(args)
        args = ICA_initial(args_dict=args_dict)
        # 仅生成模板，无需加载任何模型
        models = []
    elif args.attack == "SATA":
        # 仅生成 Prompt，加载本地 Vicuna 作为攻击模型（用于 mask 生成）
        args_dict = vars(args)
        args, attack_model, attack_tokenizer = SATA_initial(
            args_dict=args_dict, device=device
        )
        models = [attack_model, attack_tokenizer]
    elif args.attack in [
        "Actorattack",
        "Cipher",
        "CodeChameleon",
        "FlipAttack",
        "FuzzLLM",
        "Template",
        "Translate",
        "SensitivePinyin",
        "NoiseInjection",
    ]:
        models = []
    else:
        raise NameError

    # 2. 执行测试
    all_output = test(
        goals, targets, languages, models, device, args, all_output=all_output
    )

    # 3. 清理工作 (针对 DrAttack)
    if args.attack == "DrAttack":
        DrAttack_stop(worker=worker)
    # 4. 通用清理：尝试释放 GPU 显存（包括具备 shutdown 的封装类）
    try:
        import gc  # 延迟导入，避免顶层依赖
        import torch
        # 先优雅关闭具备 shutdown() 的模型封装（如 AdvPrompter/AmpleGCG/ReNeLLM）
        try:
            if isinstance(models, (list, tuple)):
                for m in models:
                    if hasattr(m, "shutdown"):
                        try:
                            m.shutdown()
                        except Exception:
                            pass
        except Exception:
            pass
        # 显式删除模型引用，释放 CUDA 缓存
        del models
        torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass
    return all_output


def main(args=None):
    """
    主函数：程序的入口点。
    负责设置随机种子、加载数据、处理断点续传、运行测试、执行 SmoothLLM 评估以及 Agent 评估。
    """
    # 兼容：若未传入或传入的是 dict/对象，则构建/合并为 argparse.Namespace
    if args is None:
        args = initialize_args()
    elif not isinstance(args, argparse.Namespace):
        # 不读取命令行，仅用默认值 + 外部覆盖生成 Namespace
        args = initialize_args(overrides=args, parse_cli=False)

    # 1. 默认设置
    set_random_seed(args.random_seed)
    # 统一默认实验名，避免路径拼接传入 None
    if not getattr(args, "exp_name", None):
        args.exp_name = "default"
    target_model_path = args.target_model_path
    args.template_name = get_template_name(target_model_path)
    args.timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M_%S")
    print("\n\ntarget_model_path", target_model_path, "\n\n")
    device = "cuda:{}".format(args.device_id)

    # 2. 加载数据 (load data)
    instructions_path = args.instructions_path
    # load_prompts: 加载完整表数据（支持 xlsx/csv，已做空值默认填充）
    data_table = load_prompts(instructions_path)
    # 若指定了 sample_size，则在完整数据上做一次随机采样
    try:
        sample_size = max(0, int(getattr(args, "sample_size", 0)))
    except Exception:
        sample_size = 0
    total_rows = len(data_table)
    if sample_size > 0 and total_rows > 0:
        if sample_size >= total_rows:
            print(
                f"[Sampling] sample_size={sample_size} >= total_rows={total_rows}, 使用全部数据，不再额外采样。",
                flush=True,
            )
        else:
            # 使用 random_seed 保证可复现
            data_table = (
                data_table.sample(n=sample_size, random_state=args.random_seed)
                .reset_index(drop=True)
            )
            print(
                f"[Sampling] 从 {total_rows} 条指令中随机采样 {sample_size} 条用于本次攻击（random_seed={args.random_seed}）。",
                flush=True,
            )
    # 从表中提取 goals 与 targets（兼容多种列名；缺失时以空字符串兜底）
    col_map = {c.lower(): c for c in data_table.columns}
    goal_candidates = ["goal", "seed_prompt", "query", "prompt"]
    target_candidates = ["target", "target_response"]
    lang_candidates = ["language", "lang"]

    def _pick_col(candidates: List[str]):
        for name in candidates:
            if name in col_map:
                return col_map[name]
        return None

    goal_col = _pick_col(goal_candidates)
    target_col = _pick_col(target_candidates)
    lang_col = _pick_col(lang_candidates)

    goals = (
        data_table[goal_col].astype(str).tolist()
        if goal_col is not None
        else [""] * len(data_table)
    )
    targets = (
        data_table[target_col].astype(str).tolist()
        if target_col is not None
        else [""] * len(data_table)
    )
    # 加载 language 字段（兼容 language/lang）
    languages_table = (
        data_table[lang_col].astype(str).tolist()
        if lang_col is not None
        else ["CHINESE_SIMPLIFIED"] * len(data_table)
    )

    # 对非法 / 未知语言名做兜底：
    # - 允许大小写无关（如 "english" / "English"）
    # - 若不在已知枚举中，则默认使用 CHINESE_SIMPLIFIED，避免 KeyError: 'UNKNOWN'
    normalized_languages = []
    for raw_lang in languages_table:
        if raw_lang is None:
            normalized_languages.append("CHINESE_SIMPLIFIED")
            continue
        name = str(raw_lang).strip().upper()
        if name == "":
            normalized_languages.append("CHINESE_SIMPLIFIED")
        elif name in Language.__members__:
            normalized_languages.append(name)
        else:
            # 未知语言名统一兜底为中文简体
            normalized_languages.append("CHINESE_SIMPLIFIED")

    languages = [Language[name] for name in normalized_languages]

    # 3. 处理数据分割 (Data Split) - 如果需要并行跑多个任务
    if args.data_split:
        print("Find data_split is True, split the data")
        args.start_index = (
            len(goals) // args.data_split_total_num
        ) * args.data_split_idx
        args.end_index = (len(goals) // args.data_split_total_num) * (
            args.data_split_idx + 1
        )
    else:
        args.start_index = 0
        args.end_index = len(goals)

    # 4. 测试准备 (断点续传逻辑)
    all_output = []
    args.test_data_idx = max(args.start_index, 0)

    # 如果设置了 resume_exp，尝试从上次保存的文件中加载进度
    if args.resume_exp:
        if args.data_split:
            new_start_idx, new_timestamp = load_test_from_file_split(args)
            args.test_data_idx = new_start_idx
            if len(new_timestamp) > 0:
                args.timestamp = new_timestamp
            print(
                f"Load the progress successfully, start from the index: {args.test_data_idx}; Current timestamp: {args.timestamp}"
            )
        else:
            all_output, new_timestamp = load_test_from_file(args)
            if len(all_output) == 0:
                print("Load the data failed, start from the beginning")
                print(f"Start from the index: {args.test_data_idx}")
            else:
                args.test_data_idx = all_output[-1]["data_id"] + 1
                if len(new_timestamp) > 0:
                    args.timestamp = new_timestamp
                print(
                    f"Load the data successfully, start from the index: {args.test_data_idx}; Current timestamp: {args.timestamp}"
                )

    # 5. 开始运行主测试流程，并统计时间与样本数
    existing_before_run = len(all_output)
    start_time = time.time()
    all_output = run(
        goals, targets, languages, target_model_path, device, args, all_output
    )
    elapsed = time.time() - start_time
    new_samples = max(0, len(all_output) - existing_before_run)

    # 6. SmoothLLM 防御效果测试 (可选)
    # 如果 defense_type 是 smoothLLM，需要进行后处理评估
    if args.defense_type == "smoothLLM" and args.attack != "DrAttack":
        final_all_output = test_smoothLLM(all_output, args)
    else:
        print(f"""\n{'=' * 36}\nNo SmoothLLM Test\n{'=' * 36}\n""")
        final_all_output = all_output

    # 7. Agent 评估 (可选)
    # 使用 GPT-4 或其他 Agent 来判断最终的攻击是否成功
    if args.agent_evaluation:
        if args.data_split:
            final_all_output = load_split_file_whole(args)
            if len(final_all_output) != len(goals):
                print(
                    "Find the final_all_output is not equal to the goals, skip the agent evaluation"
                )
                return
        else:
            print(
                "Agent evaluation: data_split is False, use in-memory results for evaluation"
            )
        if not args.agent_recheck and args.resume_exp:
            print("Find resume_exp is True, check whether need to do agent evaluation")
            if final_all_output[-1]["is_JB_Agent"] != "None":
                print(f"""\n{'*' * 36}\nSkip the agent evaluation\n{'*' * 36}\n""")
                save_test_to_file(args=args, instructions=final_all_output)
                return
            else:
                print("Start the agent evaluation")
        elif args.agent_recheck:
            print("Find agent_recheck is True, start the agent evaluation")
        else:
            print("Start the agent evaluation")
        print(f"""\n{'=' * 36}\nAgent Evaluation\n{'=' * 36}\n""")
        final_all_output = agent_evaluation(args=args, data=final_all_output)
        save_test_to_file(args=args, instructions=final_all_output)
        print(f"""\n{'=' * 36}\nFinish Agent Evaluation\n{'=' * 36}\n""")

    # 8. 导出扩展结果表（基于原表 + 新增字段）
    try:
        export_results_table(args=args, base_table=data_table, results=final_all_output)
    except Exception as e:
        print("Export results table failed:", e)

    # 9. 展开保存全部生成的 prompts（部分方法）
    try:
        if args.attack in ["AdvPrompter", "AmpleGCG", "ReNeLLM", "SensitivePinyin", "NoiseInjection"]:
            print(f"Saving exploded results for {args.attack}...")
            exploded_data = []
            for record in final_all_output:
                prompts = record.get("all_generated_prompts", [])
                if not prompts:
                    prompts = [record.get("adv_prompt", "")]
                for p in prompts:
                    new_row = record.copy()
                    new_row["jailbreak_prompt"] = p
                    if "all_generated_prompts" in new_row:
                        del new_row["all_generated_prompts"]
                    exploded_data.append(new_row)
            import pandas as pd

            save_path = os.path.join(
                args.save_result_path, f"{args.attack}_exploded_{args.timestamp}.csv"
            )
            df_ex = pd.DataFrame(exploded_data)
            df_ex.to_csv(save_path, index=False, encoding="utf-8-sig")
            print(f"Saved to {save_path}")
    except Exception as e:
        print("Exploded save failed:", e)

    # 10. 打印 & 记录攻击耗时信息，便于统一统计
    try:
        total_seconds = max(0, int(round(elapsed)))
        m, s = divmod(total_seconds, 60)
        print(
            f"""\n{'=' * 36}\nAttack timing summary\n{'=' * 36}\n"""
            f"Attack method: {getattr(args, 'attack', 'UNKNOWN')}\n"
            f"New samples in this run: {new_samples}\n"
            f"Elapsed time: {m} 分 {s} 秒 (≈ {elapsed:.2f} s)\n"
            f"""{'=' * 36}\n"""
        )
        # 追加写入日志文件，便于后续统一汇总
        try:
            log_dir = os.path.join(os.path.dirname(__file__), "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "attack_timing.log")

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{datetime.datetime.now().isoformat()} | "
                    f"attack={getattr(args, 'attack', 'UNKNOWN')} | "
                    f"new_samples={new_samples} | "
                    f"elapsed_sec={elapsed:.3f} | "
                    f"exp_name={getattr(args, 'exp_name', '')}\n"
                )
        except Exception:
            # 日志写入失败不影响主流程
            pass
    except Exception:
        # 统计逻辑出现问题时不影响主流程
        pass

    return final_all_output


if __name__ == "__main__":
    # 初始化参数
    args = initialize_args()
    # 打印参数
    args_dict = vars(args)
    print(args_dict)
    # 执行主函数
    main(args)
