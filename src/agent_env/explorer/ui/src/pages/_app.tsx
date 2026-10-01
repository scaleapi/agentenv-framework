import '../styles/globals.css';
import '@radix-ui/themes/styles.css';
import { Theme } from '@radix-ui/themes';
import type { AppProps } from 'next/app';
import { Inter, Geist_Mono } from 'next/font/google';
import Head from 'next/head';
import BackendStatusBanner from '../components/backend-status-banner';

/*
 * agent-env explorer typography: Inter for UI, Geist Mono for data/code.
 * next/font self-hosts both at build time (works with the static export), and
 * exposes each as a CSS variable consumed by Tailwind's font stacks
 * (`src/styles/tokens.ts`) and by Radix Themes (`--default-font-family` /
 * `--code-font-family`, wired in `globals.css`). The variables are attached to
 * the <Theme> wrapper below, which is the outermost element of every page.
 */
const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});
const geistMono = Geist_Mono({
  subsets: ['latin'],
  variable: '--font-geist-mono',
  display: 'swap',
});

function MyApp({ Component, pageProps }: AppProps) {
  return (
    <Theme
      className={`${inter.variable} ${geistMono.variable}`}
      appearance="light"
      accentColor="gray"
      grayColor="gray"
      radius="small"
      scaling="100%"
    >
      <Head>
        {/*
          Adaptive SVG favicon: the brand mark switches its fills with an
          internal `@media (prefers-color-scheme)` query, so the icon re-renders
          live when the OS/tab-bar theme changes (light fills on dark chrome,
          dark fills on light). This is the only approach browsers re-evaluate on
          scheme change — per-<link> `media` on raster icons is cached/hijacked by
          the `sizes="any"` .ico and gets stuck showing one glyph.
        */}
        <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
        {/* Legacy fallback for browsers without SVG favicon support (e.g. Safari).
            No `sizes="any"` so it never outranks the SVG in modern browsers. */}
        <link rel="icon" href="/favicon.ico" sizes="32x32" />
        <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
        <link rel="manifest" href="/site.webmanifest" />
      </Head>
      <Component {...pageProps} />
      <BackendStatusBanner />
    </Theme>
  );
}
export default MyApp;
