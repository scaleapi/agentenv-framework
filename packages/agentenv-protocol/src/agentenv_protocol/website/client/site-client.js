// The browser client of a site on agentenv_protocol.website: JSON calls to the site's API, the universe's time from
// X-Universe-Now, and the signed-in user. It has no dependencies. Paths are relative to the page, so a site works at
// "/" and under the gateway's "/website/<name>/" alike: a leading "/" is dropped.

let offsetMs = 0;

// An answer other than success: `status` (0 when the server couldn't be reached), and the site's `code` and `error`.
export class ApiError extends Error {
  constructor(status, code, error, body) {
    super(error);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.error = error;
    this.body = body;
  }
}

// The universe's time: the X-Universe-Now of the latest answer, extrapolated at real speed since it came. An idle
// page keeps it right with followUniverseClock().
export function universeNow() {
  return new Date(Date.now() + offsetMs);
}

// Read X-Universe-Now from `path`, a GET of the site's own that changes nothing, such as a list of its users, every
// `everyMs`, so universeNow() follows a clock that stops or runs at another speed while the page makes no calls.
// Returns a function that stops it.
export function followUniverseClock({ path, everyMs = 5000 } = {}) {
  if (!path) throw new TypeError("followUniverseClock needs the path of a GET the site serves");
  const id = setInterval(() => fetch(relative(path), { credentials: "same-origin" }).then(follow).catch(() => {}), everyMs);
  return () => clearInterval(id);
}

function follow(response) {
  const at = Date.parse(response.headers.get("X-Universe-Now") || "");
  if (!Number.isNaN(at)) offsetMs = at - Date.now();
}

function relative(path) {
  return String(path).replace(/^\/+/, "");
}

// Call the site's API: `body` is sent as JSON; the answer's JSON is returned, or an ApiError thrown.
export async function api(path, { method = "GET", body, headers = {} } = {}) {
  const url = relative(path);
  let response;
  try {
    response = await fetch(url, {
      method,
      credentials: "same-origin",
      headers: body === undefined ? headers : { "Content-Type": "application/json", ...headers },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (cause) {
    throw new ApiError(0, "network_error", `${method} ${url}: ${cause?.message || cause}`, null);
  }
  follow(response);
  let text;
  try {
    text = await response.text();
  } catch (cause) {
    throw new ApiError(response.status, "network_error", `${method} ${url}: ${cause?.message || cause}`, null);
  }
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  if (!response.ok) {
    const error = data && typeof data === "object" && data.error ? data.error : `${method} ${url}: ${response.status}`;
    const code = data && typeof data === "object" && data.code ? data.code : "http_error";
    throw new ApiError(response.status, code, error, data);
  }
  return data;
}

// Who the browser is signed in as: {user, role, signed_in}. A browser with no session is signed in as the site's
// acting user.
export function session() {
  return api("api/session");
}

// Sign in as `user` through the site's own sign-in route, `api/sign-in` unless `path` names another.
export function signIn(user, { path = "api/sign-in" } = {}) {
  return api(path, { method: "POST", body: { user } });
}

export function signOut() {
  return api("api/session", { method: "DELETE" });
}
