from src.patch_predict import predict_landcover


def detect_changes(image1_path, image2_path):
    """
    Compare two satellite images patch-by-patch.

    Returns:
        changes (list)
    """

    image1 = predict_landcover(image1_path)
    image2 = predict_landcover(image2_path)

    changes = []

    for patch1, patch2 in zip(image1, image2):

        if patch1["class"] != patch2["class"]:

            changes.append(
                {
                    "x": patch1["x"],
                    "y": patch1["y"],
                    "before": patch1["class"],
                    "after": patch2["class"],
                    "confidence_before": patch1["confidence"],
                    "confidence_after": patch2["confidence"],
                }
            )

    return changes


if __name__ == "__main__":

    changes = detect_changes(
        "2018.jpg",
        "2026.jpg"
    )

    print(f"Total changes detected: {len(changes)}")

    for change in changes:
        print(change)