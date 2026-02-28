from typing import Set, List
from itertools import combinations

def is_sigma_algebra(omega: Set, E: List[Set]) -> bool:
    """
    Check if E is a sigma-algebra on the sample space omega.
    
    Args:
        omega (Set): The sample space
        E (List[Set]): Collection of sets
        
    Returns:
        bool: True if E is a sigma-algebra on omega, False otherwise
    """
    phi = omega - omega

    if not omega in E: return False
    if not phi in E: return False

    for A in E: 
        if not (omega - A) in E : return False
    
    union = set()
    for A in E:
        union = union.union(A)
    if not union in E: return False

    return True


def complete_sigma_algebra(omega: Set, E: List[Set]) -> List[Set]:
    """
    Returns the smallest sigma-algebra on omega that contains all sets in E. If a set in E is not a subset of omega, it must not be included in the resulting sigma-algebra.
    
    Args:
        omega (Set): The sample space
        E (List[Set]): Collection of sets
        
    Returns:
        List[Set]: The smallest sigma-algebra on omega that contains all sets in E (that are subsets of omega).
    """

    smallest_sigma_algebra, valid_sets, phi = [], [], omega-omega

    for A in E:
        if A.issubset(omega): valid_sets.append(A)

    smallest_sigma_algebra.extend(valid_sets)
    
    if not omega in valid_sets: smallest_sigma_algebra.append(omega)
    if not phi in valid_sets: smallest_sigma_algebra.append(phi)

    for A in valid_sets:
        Ac = omega - A
        if Ac not in smallest_sigma_algebra: smallest_sigma_algebra.append(Ac)

    print(E)
    print("Smallest Sigma Algebra:", smallest_sigma_algebra)
    
    return smallest_sigma_algebra


def run_tests():
    # TODO: Implement at least 5 tests for `is_sigma_algebra`.
    # Simply use `assert` statements to check the correctness of your implementation.
    omega = {1, 2, 3, 4, 5, 6}
    F1 = [set(), omega]
    F2 = [set(), {1, 3, 5}, {2, 4, 6}, {1, 2, 3, 4, 5, 6}] 
    F3 = [set(), {1, 2}, {3, 4}, {5, 6}, {3, 4, 5, 6}, {1, 2, 5, 6}, {1, 2, 3, 4}, {1, 2, 3, 4, 5, 6}]
    F4 = [set(), {1}, {2, 3, 4, 6}, {1, 2, 3, 4, 5, 6}]
    F5 = [set(sub_set) for range in range(len(omega) + 1) for sub_set in combinations(omega, range)]
    assert is_sigma_algebra(omega, F1) is True
    assert is_sigma_algebra(omega, F2) is True
    assert is_sigma_algebra(omega, F3) is True
    assert is_sigma_algebra(omega, F4) is False
    assert is_sigma_algebra(omega, F5) is True
    print('All tests passed for is_sigma_algebra.')

    # TODO: Implement at least 5 tests for `complete_sigma_algebra`.
    # Simply use `assert` statements to check the correctness of your implementation.
    omega = {1, 2, 3, 4, 5, 6}
    F1 = [{1, 2}, {4, 8}]
    F2 = [{1}]
    F3 = [{3, 4, 5}]
    F4 = [set()]
    F5 = [omega]
    assert is_sigma_algebra(omega, complete_sigma_algebra(omega, F1)) is True
    assert is_sigma_algebra(omega, complete_sigma_algebra(omega, F2)) is True
    assert is_sigma_algebra(omega, complete_sigma_algebra(omega, F3)) is True
    assert is_sigma_algebra(omega, complete_sigma_algebra(omega, F4)) is True
    assert is_sigma_algebra(omega, complete_sigma_algebra(omega, F5)) is True
    print('All tests passed for complete_sigma_algebra.')


if __name__ == '__main__':
    run_tests()