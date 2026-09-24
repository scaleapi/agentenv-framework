// pptx-preview ships no type declarations. Minimal ambient shim covering the
// `init(container, options).preview(bytes)` API we use in ArtifactPptxPreview.
declare module 'pptx-preview' {
  interface PptxPreviewer {
    preview(data: ArrayBuffer | Uint8Array): void | Promise<void>;
  }
  export function init(
    container: HTMLElement,
    options?: { width?: number; height?: number },
  ): PptxPreviewer;
}
