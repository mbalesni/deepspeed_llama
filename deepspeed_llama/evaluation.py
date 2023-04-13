from typing import Dict


# NOTE: this is an example function, replace this with you own metric
def evaluate_completions_exact_match(args, completions, targets, case_sensitive=False) -> Dict:
    '''Compute accuracy of completions using exact-match.
    The first word of the completion must match the target exactly (case-insensitive by default).
    e.g. completion " World is vast" with target "world" is correct
    '''
    n_correct = 0
    is_correct_list = []

    for completion, target in zip(completions, targets):
        target = target.strip()
        test_str = completion.strip()
        test_str = test_str.lower() if not case_sensitive else test_str
        target_str = target.lower() if not case_sensitive else target
        correct = test_str.startswith(target_str)
        is_correct_list.append(correct)
        if correct:
            n_correct += 1

    accuracy = n_correct / len(completions)
    if args.verbose:
        print()

    results = {
        'accuracy': accuracy,
        'is_correct_list': is_correct_list
    }
    return results
