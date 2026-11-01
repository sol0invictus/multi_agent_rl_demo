// Render multi-agent-rl.md to a standalone HTML page in the sunnyguha.com blog style.
//
//     NODE_PATH=/home/webpage/node_modules node blog/build.mjs
//
// Uses the same Markdown pipeline as the Astro site (GFM, smartypants, remark-math -> rehype-katex,
// Shiki github-light with wrap) so the output matches /blog/<slug> there. style.css is a snapshot of
// the site's src/styles/global.css. Writes multi-agent-rl.html next to the post, so the figure
// paths (./multi-agent-rl/...) resolve unchanged.
import { readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(join(process.env.NODE_PATH || '/home/webpage/node_modules', 'x.js'));
const imp = (name) => import(require.resolve(name));
const { unified } = await imp('unified');
const remarkParse = (await imp('remark-parse')).default;
const remarkGfm = (await imp('remark-gfm')).default;
const remarkMath = (await imp('remark-math')).default;
const remarkSmartypants = (await imp('remark-smartypants')).default;
const remarkRehype = (await imp('remark-rehype')).default;
const rehypeKatex = (await imp('rehype-katex')).default;
const rehypeStringify = (await imp('rehype-stringify')).default;
const GithubSlugger = (await imp('github-slugger')).default;
const { createHighlighter } = await imp('shiki');

const HERE = dirname(fileURLToPath(import.meta.url));
const SLUG = 'multi-agent-rl';
const SITE = 'https://www.sunnyguha.com';
const KATEX_CSS = 'katex/katex.min.css';

const src = readFileSync(join(HERE, `${SLUG}.md`), 'utf8');
const [, fm, body] = src.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
const meta = Object.fromEntries(fm.split('\n').map((l) => {
  const i = l.indexOf(':');
  return [l.slice(0, i).trim(), l.slice(i + 1).trim().replace(/^"(.*)"$/, '$1')];
}));

const highlighter = await createHighlighter({ themes: ['github-light'], langs: ['bash', 'python', 'text'] });
const text = (n) => (n.type === 'text' ? n.value : (n.children || []).map(text).join(''));

function rehypeAstroLike() {
  return (tree) => {
    const slugger = new GithubSlugger();
    const walk = (node) => {
      (node.children || []).forEach((child, i) => {
        if (child.type !== 'element') return;
        if (/^h[1-6]$/.test(child.tagName)) child.properties.id = slugger.slug(text(child));
        if (child.tagName === 'pre' && child.children[0]?.tagName === 'code') {
          const code = child.children[0];
          const cls = (code.properties.className || []).find((c) => c.startsWith('language-'));
          const lang = cls ? cls.slice(9) : 'text';
          const hast = highlighter.codeToHast(text(code).replace(/\n$/, ''), { lang, theme: 'github-light' });
          const pre = hast.children[0];
          pre.properties.className = ['astro-code', 'github-light'];
          pre.properties.style += '; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word;';
          node.children[i] = pre;
          return;
        }
        if (child.tagName === 'img') child.properties.loading = 'lazy';
        walk(child);
      });
    };
    walk(tree);
  };
}

const html = String(await unified()
  .use(remarkParse).use(remarkGfm).use(remarkSmartypants).use(remarkMath)
  .use(remarkRehype).use(rehypeKatex).use(rehypeAstroLike).use(rehypeStringify)
  .process(body));

const date = new Date(meta.date + 'T00:00:00Z');
const fmt = date.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric', timeZone: 'UTC' });
const css = readFileSync(join(HERE, 'style.css'), 'utf8');

const page = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="description" content="${meta.description.replace(/"/g, '&quot;')}" />
    <title>${meta.title} — Sunny Guha</title>
    <link rel="stylesheet" href="${KATEX_CSS}" />
    <style>
${css}
    </style>
  </head>
  <body>
    <header class="site-header">
      <div class="site-header-inner">
        <a href="${SITE}/" class="site-title">Sunny Guha</a>
        <nav class="site-nav">
          <a href="${SITE}/">About</a>
          <a href="${SITE}/blog" class="is-active">Blog</a>
          <a href="${SITE}/projects">Projects</a>
          <a href="${SITE}/publications">Publications</a>
        </nav>
      </div>
    </header>
    <main>
      <article>
        <h1>${meta.title}</h1>
        <p style="color: var(--fg-muted); margin-top: -0.4em;">
          <time datetime="${date.toISOString()}">${fmt}</time>
        </p>
${html}
      </article>
      <hr />
      <p><a href="${SITE}/blog">← Back to blog</a></p>
    </main>
    <footer class="site-footer">
      &copy; ${date.getUTCFullYear()} Sunny Guha
    </footer>
  </body>
</html>
`;
writeFileSync(join(HERE, `${SLUG}.html`), page);
console.log(`wrote ${SLUG}.html (${(page.length / 1024).toFixed(0)} KB)`);
