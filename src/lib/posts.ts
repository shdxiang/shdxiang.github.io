import type { CollectionEntry } from "astro:content";

const DATE_PREFIX = /^\d{4}-\d{2}-\d{2}-/;

function getRawSlug(post: CollectionEntry<"posts">): string {
  return post.id.replace(/\.md$/, "");
}

export function getSlugWithoutDate(post: CollectionEntry<"posts">): string {
  const rawSlug = getRawSlug(post);
  return rawSlug.replace(DATE_PREFIX, "");
}

export function getPostPath(post: CollectionEntry<"posts">): string {
  const yyyy = String(post.data.date.getFullYear());
  const mm = String(post.data.date.getMonth() + 1).padStart(2, "0");
  const dd = String(post.data.date.getDate()).padStart(2, "0");
  return `/posts/${yyyy}/${mm}/${dd}/${getSlugWithoutDate(post)}/`;
}

export function formatDate(date: Date): string {
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "short",
    day: "numeric"
  }).format(date);
}
