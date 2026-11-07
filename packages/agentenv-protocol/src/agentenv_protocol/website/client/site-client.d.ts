/** An answer other than success: `status` (0 when the server couldn't be reached), and the site's `code` and `error`. */
export declare class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly error: string;
  readonly body: unknown;
  constructor(status: number, code: string, error: string, body: unknown);
}

export interface ApiOptions {
  method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE" | string;
  body?: unknown;
  headers?: Record<string, string>;
}

export interface Session {
  user: string | null;
  role: string;
  signed_in: boolean;
}

/** The universe's time: the X-Universe-Now of the latest answer, extrapolated at real speed since it came. */
export declare function universeNow(): Date;
/** Read X-Universe-Now from `path`, a GET of the site's own that changes nothing, every `everyMs` (5000), so universeNow() follows a clock that stops or runs at another speed; returns a function that stops it. */
export declare function followUniverseClock(options: { path: string; everyMs?: number }): () => void;

/** Call the site's API at `path`, relative to the page; returns the answer's JSON or throws an ApiError. */
export declare function api<T = any>(path: string, options?: ApiOptions): Promise<T>;

/** Who the browser is signed in as; a browser with no session is signed in as the site's acting user. */
export declare function session(): Promise<Session>;

/** Sign in as `user` through the site's own sign-in route, `api/sign-in` unless `path` names another. */
export declare function signIn<T = { user: string }>(user: string, options?: { path?: string }): Promise<T>;

export declare function signOut(): Promise<Session>;
