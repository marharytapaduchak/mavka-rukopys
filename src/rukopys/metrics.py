"""Edit distance and character error rate metrics."""

def edit_distance(a, b):
    """Compute the Levenshtein distance between two sequences."""
    previous = list(range(len(b) + 1))

    for i, x in enumerate(a, 1):
        current = [i]

        for j, y in enumerate(b, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (x != y),
                )
            )

        previous = current

    return previous[-1]


def cer_summary(rows):
    """Compute corpus and mean regional CER for non-empty reference texts."""
    scored = [r for r in rows if r.get("reference", "")]

    if not scored:
        return {
            "n": 0,
            "corpus_raw_cer": None,
            "mean_region_raw_cer": None,
        }

    return {
        "n": len(scored),
        "corpus_raw_cer": (
            sum(edit_distance(r["reference"], r["prediction"]) for r in scored)
            / sum(len(r["reference"]) for r in scored)
        ),
        "mean_region_raw_cer": (
            sum(
                edit_distance(r["reference"], r["prediction"]) / len(r["reference"]) for r in scored
            )
            / len(scored)
        ),
    }
