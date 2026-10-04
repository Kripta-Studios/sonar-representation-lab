from analysis.diagnostics import analyze, match, size_bin


def test_greedy_duplicates_misses_negatives_and_thresholds():
    gt = {
        "images": [
            {"id": 1, "file_name": "a_1.jpg", "width": 448, "height": 448},
            {"id": 2, "file_name": "b_1.jpg", "width": 448, "height": 448},
        ],
        "categories": [{"id": 1}],
        "annotations": [
            {"id": 4, "image_id": 1, "category_id": 1, "bbox": [0, 0, 8, 8]},
            {"id": 5, "image_id": 1, "category_id": 1, "bbox": [100, 100, 8, 8]},
        ],
    }
    predictions = [
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 8, 8], "score": 0.9},
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 8, 8], "score": 0.8},
        {"image_id": 2, "category_id": 1, "bbox": [0, 0, 8, 8], "score": 0.2},
    ]
    result = analyze(gt, predictions, (0.1, 0.5))
    low, high = result["operating_points"]
    assert (low["tp"], low["fp"], low["fn"], low["negative_false_positives"]) == (1, 2, 1, 1)
    assert (high["tp"], high["fp"], high["fn"], high["negative_false_positives"]) == (1, 1, 1, 0)
    assert high["false_positive_types"] == {"duplicate": 1, "localization": 0, "background": 0}
    assert high["size_bins"]["8-16px"]["recall"] == 0.5
    assert result["error_examples"][0]["annotation_id"] == 5
    assert sum(c["fn"] for c in high["clips"].values()) == high["fn"]


def test_fixed_reference_size_and_category_matching():
    annotation = {"category_id": 1, "bbox": [0, 0, 16, 16]}
    assert size_bin(annotation, {"width": 896, "height": 500}) == "8-16px"
    found, decisions = match([annotation], [{"category_id": 2, "bbox": [0, 0, 16, 16], "score": 1.0}])
    assert not found and decisions == ["background"]
