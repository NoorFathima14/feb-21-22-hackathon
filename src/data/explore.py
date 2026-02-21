import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def get_dataset_paths():
    """
    Returns dataset path depending on environment.
    Kaggle environment path is detected automatically.
    """

    kaggle_path = "/kaggle/input/cifake-real-and-ai-generated-synthetic-images"
    local_path = os.path.join("data", "cifake")

    if os.path.exists(kaggle_path):
        print("Running in Kaggle environment.")
        dataset_dir = kaggle_path
    elif os.path.exists(local_path):
        print("Running in local environment.")
        dataset_dir = local_path
    else:
        raise FileNotFoundError(
            "Dataset not found. Ensure CIFAKE is placed correctly."
        )

    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")

    return train_dir, test_dir


def show_examples(base_dir, title):
    classes = ['FAKE', 'REAL']

    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    fig.suptitle(title, fontsize=12)

    for i, cls in enumerate(classes):
        cls_dir = os.path.join(base_dir, cls)
        sample_files = random.sample(os.listdir(cls_dir), 2)

        for j, img_name in enumerate(sample_files):
            img_path = os.path.join(cls_dir, img_name)
            img = mpimg.imread(img_path)

            ax = axes[i * 2 + j]
            ax.imshow(img)
            ax.set_title(cls)
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def print_dataset_stats(train_dir, test_dir):
    classes = ['FAKE', 'REAL']

    print("\nDataset Statistics:")
    print("-" * 40)

    for split_name, split_dir in [("Train", train_dir), ("Test", test_dir)]:
        print(f"\n{split_name} Split:")
        total = 0
        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)
            count = len(os.listdir(cls_dir))
            print(f"{cls}: {count}")
            total += count
        print(f"Total {split_name} Samples: {total}")

    print("-" * 40)


if __name__ == "__main__":
    train_dir, test_dir = get_dataset_paths()

    print("Train Directory:", train_dir)
    print("Test Directory:", test_dir)

    print_dataset_stats(train_dir, test_dir)

    show_examples(train_dir, "Train Dataset Samples")
    show_examples(test_dir, "Test Dataset Samples")