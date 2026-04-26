import type { CollectionEntry } from "astro:content";
import { createHash } from "node:crypto";
import { basename } from "node:path";
import { withBase } from "./paths";

function getPostFilename(post: CollectionEntry<"posts">): string {
  if (post.filePath) {
    return basename(post.filePath);
  }
  return post.id.endsWith(".md") ? post.id : `${post.id}.md`;
}

export function getPostHash(post: CollectionEntry<"posts">): string {
  return createHash("md5").update(getPostFilename(post)).digest("hex");
}

export function getPostPath(post: CollectionEntry<"posts">): string {
  return withBase(`/posts/${getPostHash(post)}/`);
}

export function formatDate(date: Date): string {
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "short",
    day: "numeric",
    timeZone: "Asia/Shanghai"
  }).format(date);
}
