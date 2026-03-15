export default function MarkdownCard({ content }) {
  const html = String(content || '')
    .replace(/^### (.*)$/gm, '<h3>$1</h3>')
    .replace(/^## (.*)$/gm, '<h2>$1</h2>')
    .replace(/^# (.*)$/gm, '<h1>$1</h1>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/^- (.*)$/gm, '<li>$1</li>')
    .replace(/^• (.*)$/gm, '<li>$1</li>')
    .replace(/(?:\r\n|\r|\n){2,}/g, '</p><p>')
    .replace(/\n/g, '<br />')

  const wrapped = html
    .replace(/(<li>.*?<\/li>)/gs, '<ul>$1</ul>')
    .replace(/<ul><ul>/g, '<ul>')
    .replace(/<\/ul><\/ul>/g, '</ul>')

  return <div className="markdown-card" dangerouslySetInnerHTML={{ __html: `<p>${wrapped}</p>` }} />
}
