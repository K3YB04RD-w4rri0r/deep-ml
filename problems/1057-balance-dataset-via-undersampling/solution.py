def balance_undersample(data: list) -> list:
    """
    Undersample the majority classes so all classes have the same number of
    samples equal to the minority class count.

    data: list of (sample, label) tuples
    Returns: list of (sample, label) tuples, order-preserving
    """
    if data == []:
        return  []

    class_counts = {}
    for p,c in data:
        class_counts[c] = class_counts.get(c, 0) + 1
    min_count = min(class_counts.values())

    out = []
    taken = {c : 0 for c in class_counts}
    for s,c in data:
        if taken[c] < min_count:
            taken[c] += 1
            out.append((s,c))

    return out

        
    
    



    
