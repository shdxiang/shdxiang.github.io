import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import React from "react";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import rehypeSlug from "rehype-slug";
import rehypeAutolinkHeadings from "rehype-autolink-headings";

type Props = {
  content: string;
};

function getLang(children: React.ReactNode): string {
  const child = React.Children.toArray(children)[0];
  if (!React.isValidElement(child)) return "";
  const cls = (child.props as Record<string, unknown>).className;
  if (typeof cls !== "string") return "";
  const m = cls.match(/language-(\S+)/);
  return m ? m[1] : "";
}

const components: Components = {
  pre({ children, ...props }) {
    const lang = getLang(children);
    return (
      <div className="code-block">
        <div className="code-block-bar">
          {lang && <span className="code-block-lang">{lang}</span>}
          <button type="button" className="copy-code-btn" aria-label="Copy code" title="Copy code">
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path
                fill="currentColor"
                d="M16 1H6a2 2 0 0 0-2 2v12h2V3h10V1Zm3 4H10a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2Zm0 16H10V7h9v14Z"
              />
            </svg>
          </button>
        </div>
        <pre {...props}>{children}</pre>
      </div>
    );
  }
};

export default function MarkdownRenderer({ content }: Props) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[
        rehypeKatex,
        rehypeHighlight,
        rehypeSlug,
        [rehypeAutolinkHeadings, { behavior: "append" }],
      ]}
      components={components}
    >
      {content}
    </ReactMarkdown>
  );
}
