import { safeArtifactHref } from "../schemas/contracts";

export function ArtifactList({ artifacts }: { artifacts: string[] }) {
  const unique = [...new Set(artifacts)];
  return (
    <section className="panel" aria-labelledby="artifact-title">
      <div className="panel-kicker">EVIDENCE</div>
      <h2 id="artifact-title">Artifacts / 结果文件</h2>
      <ul className="artifact-list">
        {unique.map((artifact) => {
          const href = safeArtifactHref(artifact);
          const name = artifact.split("/").at(-1) ?? artifact;
          return <li key={artifact}>{href ? <a href={href}>{name}<span aria-hidden="true"> ↗</span></a> : <span>{name} · UNSAFE LINK</span>}</li>;
        })}
      </ul>
    </section>
  );
}
