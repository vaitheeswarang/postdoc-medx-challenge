import os
import json
import argparse
from tqdm import tqdm
import numpy as np

from medmnist import PneumoniaMNIST


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--num_images", type=int, default=10)
    p.add_argument("--out_dir", type=str, default="reports/task2")
    p.add_argument("--use_4bit", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=200)
    return p.parse_args()


def build_prompt():
    # Keep prompt simple and consistent; label-aware prompting can be added later for analysis
    return (
        "Describe this chest X-ray in 2-4 sentences. "
        "Mention lung fields, focal opacities/consolidation, pleural effusion if present, "
        "and end with a cautious impression."
    )


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    images_dir = os.path.join(args.out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Lazy import to avoid loading the big model when only checking dataset
    from task2_report_generation.vlm import VLM

    dataset = PneumoniaMNIST(split=args.split, download=True)

    vlm = VLM(model_id=args.model_id, use_4bit=args.use_4bit)

    results = []
    prompt = build_prompt()

    n = min(args.num_images, len(dataset))
    for idx in tqdm(range(n)):
        img, label = dataset[idx]
        label = int(np.asarray(label).reshape(-1)[0])

        # Save the original image for the report (keep as RGB 224 for easier viewing)
        img_rgb = img.convert("RGB").resize((224, 224))
        img_name = f"img_{idx:04d}_label_{label}.png"
        img_path = os.path.join(images_dir, img_name)
        img_rgb.save(img_path)

        report = vlm.generate(img, prompt=prompt, max_new_tokens=args.max_new_tokens)

        results.append(
            {
                "index": idx,
                "ground_truth_label": label,
                "image_file": f"images/{img_name}",
                "prompt": prompt,
                "report": report,
            }
        )

    # Save JSON
    json_path = os.path.join(args.out_dir, "reports_task2.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save Markdown
    md_path = os.path.join(args.out_dir, "reports_task2.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Task-2 Sample Reports\n\n")
        for r in results:
            f.write(f"## Image {r['index']}\n")
            f.write(f"**Ground Truth Label:** {r['ground_truth_label']}\n\n")
            f.write(f"![Image {r['index']}]({r['image_file']})\n\n")
            f.write("### Prompt\n")
            f.write(f"{r['prompt']}\n\n")
            f.write("### Generated Report\n")
            f.write((r["report"] or "").strip() + "\n\n")
            f.write("---\n\n")

    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")
    print(f"Images saved in: {images_dir}")


if __name__ == "__main__":
    main()
