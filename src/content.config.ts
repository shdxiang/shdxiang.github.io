import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

const posts = defineCollection({
  loader: glob({ pattern: "*.md", base: "./_posts" }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    arxiv_id: z.string().optional(),
    tags: z.array(z.string()).optional(),
    categories: z.array(z.string()).optional(),
    description: z.string().optional(),
    draft: z.boolean().optional()
  })
});

export const collections = {
  posts
};
