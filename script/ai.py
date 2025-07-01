#!/usr/bin/env python

import os
import requests

from datetime import datetime

API_KEY = os.getenv("DEEPSEEK_API_KEY")
API_URL = "https://api.deepseek.com/v1/chat/completions"


def handle_story(story: str) -> None:
    lines = [x.strip() for x in story.split("\n")]

    title = lines[0][1:-1]
    content = "\n".join(lines[2:])

    with open(f"_posts/{datetime.now().strftime('%Y-%m-%d')}-{title}.md", "w") as f:
        f.write(f"---\n")
        f.write(f"title: {title}\n")
        f.write(f"date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S +0800')}\n")
        f.write(f"---\n")
        f.write(f"\n")
        f.write(f"{content}\n")


def main() -> None:

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    old_stories = [
        x.split("-")[3].split(".")[0] for x in os.listdir("_posts") if x.endswith(".md")
    ]

    user_prompt = f"请创作一个800字左右的魔幻现实主义短篇故事, 只输出故事内容, 不要输出任何其他内容, 不要与以下故事雷同: {', '.join(old_stories)}"

    print(user_prompt)

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一位精通魔幻现实主义的作家"},
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "temperature": 0.7,
        "max_tokens": 1500,
    }

    response = requests.post(API_URL, json=data, headers=headers)
    result = response.json()

    if "choices" in result:
        story = result["choices"][0]["message"]["content"]
        handle_story(story)
    else:
        print("failed to generate: ", result.get("error"))


if __name__ == "__main__":
    main()
