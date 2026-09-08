# ============================================================
# MIT-BIH ANNOTATION LABELING
# ============================================================

# Mapping from MIT-BIH annotation symbols
# to 5 heartbeat classes.

LABEL_MAP = {
    # --------------------------------------------------------
    # N = Normal beat
    # --------------------------------------------------------
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    # --------------------------------------------------------
    # S = Supraventricular ectopic beat
    # --------------------------------------------------------
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    # --------------------------------------------------------
    # V = Ventricular ectopic beat
    # --------------------------------------------------------
    "V": "V",
    "E": "V",
    # --------------------------------------------------------
    # F = Fusion beat
    # --------------------------------------------------------
    "F": "F",
    # --------------------------------------------------------
    # Q = Unknown / other beat
    # --------------------------------------------------------
    "/": "Q",
    "f": "Q",
    "Q": "Q",
}


# ============================================================
# CONVERT ONE ANNOTATION
# ============================================================


def convert_label(symbol):
    """
    Convert an MIT-BIH annotation symbol
    into one of the five classes.

    Returns:
        Class label if the symbol is supported.
        None otherwise.
    """

    return LABEL_MAP.get(symbol)


# ============================================================
# CHECK WHETHER SYMBOL IS A HEARTBEAT
# ============================================================


def is_heartbeat(symbol):
    """
    Check whether an annotation symbol belongs
    to one of our five heartbeat classes.
    """

    return symbol in LABEL_MAP


# ============================================================
# TEST THE LABELING SYSTEM
# ============================================================


def main():

    test_symbols = [
        "N",
        "L",
        "R",
        "A",
        "a",
        "J",
        "S",
        "V",
        "E",
        "F",
        "/",
        "f",
        "Q",
        "+",
    ]

    print("MIT-BIH → 5 CLASS LABEL MAPPING")
    print("--------------------------------")

    for symbol in test_symbols:
        label = convert_label(symbol)

        if label is None:
            print(f"Symbol: {symbol:>2} → Ignored")

        else:
            print(f"Symbol: {symbol:>2} → Class: {label}")


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":
    main()
