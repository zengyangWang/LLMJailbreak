from baseline.ReNeLLM.llm import LocalReNeLLM


def load_target_models_renellm(args):
    """
    与 AmpleGCG/AdvPrompter 的 utils 风格一致：返回一个具备 generate(args, goal, language) 的对象。
    """
    test_generation_kwargs = {
        "max_tokens": args.target_max_n_tokens,
        "temperature": 0.01,
    }
    target_llm = LocalReNeLLM(
        args=args,
        test_generation_kwargs=test_generation_kwargs,
    )
    return target_llm

