//! SQL text shared by the SQLite and Postgres stores, built once per store.
use crate::record::{COLUMNS, TERMINAL};

/// Status slots of the transition statement: one per status, unused slots bound to NULL.
pub const TRANSITION_SLOTS: usize = 6;
/// Bound values of an event append: response id, sequence and payload.
const APPEND_PARAMETERS: usize = 3;

/// Every statement a store runs, written with the backend's placeholder syntax.
#[derive(Debug)]
pub struct Statements {
    pub purge: String,
    pub get: String,
    pub put: String,
    pub put_unless_status: String,
    pub put_unless_terminal: String,
    pub transition: String,
    pub status: String,
    /// [`Self::status`] holding the row lock until commit, where the backend supports it.
    pub status_locked: String,
    pub event_tail: String,
    pub append_event: String,
    pub events: String,
    pub delete: String,
}

/// Spells the bound parameters of one statement in order: `?1, ?2, ...` or `$1, $2, ...`.
struct Parameters<'a> {
    spell: &'a dyn Fn(usize) -> String,
    next: usize,
}

impl<'a> Parameters<'a> {
    fn new(spell: &'a dyn Fn(usize) -> String) -> Self {
        Self { spell, next: 1 }
    }

    fn next(&mut self) -> String {
        let spelled = (self.spell)(self.next);
        self.next += 1;
        spelled
    }

    fn list(&mut self, count: usize) -> String {
        (0..count)
            .map(|_| self.next())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl Statements {
    /// Build the statements; `placeholder(n)` spells the n-th (1-based) bound parameter.
    pub fn new(placeholder: impl Fn(usize) -> String) -> Self {
        let spell: &dyn Fn(usize) -> String = &placeholder;
        let terminal = terminal_list();
        let expired = |table: &str, parameter: String| {
            format!(
                "{table}status IN ({terminal}) AND {table}expires_at IS NOT NULL \
                 AND {table}expires_at <= {parameter}"
            )
        };
        let columns: Vec<&str> = COLUMNS
            .split(',')
            .map(str::trim)
            .chain(["expires_at"])
            .collect();
        let updates = columns
            .iter()
            .skip(1)
            .map(|column| format!("{column} = excluded.{column}"))
            .collect::<Vec<_>>()
            .join(", ");
        let mut upsert = Parameters::new(spell);
        let put = format!(
            "INSERT INTO infer_responses ({}) VALUES ({}) \
             ON CONFLICT (id) DO UPDATE SET {updates}",
            columns.join(", "),
            upsert.list(columns.len())
        );
        let unless_status = upsert.next();
        let mut get = Parameters::new(spell);
        let mut transition = Parameters::new(spell);
        let mut events = Parameters::new(spell);
        let mut append = Parameters::new(spell);
        Self {
            purge: format!(
                "DELETE FROM infer_responses WHERE {}",
                expired("", Parameters::new(spell).next())
            ),
            get: format!(
                "SELECT {COLUMNS} FROM infer_responses WHERE id = {} AND NOT ({})",
                get.next(),
                expired("", get.next())
            ),
            put_unless_status: format!("{put} WHERE infer_responses.status <> {unless_status}"),
            put_unless_terminal: format!("{put} WHERE infer_responses.status NOT IN ({terminal})"),
            put,
            transition: format!(
                "UPDATE infer_responses SET status = {}, expires_at = {} \
                 WHERE id = {} AND status IN ({}) RETURNING {COLUMNS}",
                transition.next(),
                transition.next(),
                transition.next(),
                transition.list(TRANSITION_SLOTS)
            ),
            status: format!(
                "SELECT status FROM infer_responses WHERE id = {}",
                Parameters::new(spell).next()
            ),
            status_locked: format!(
                "SELECT status FROM infer_responses WHERE id = {} FOR UPDATE",
                Parameters::new(spell).next()
            ),
            event_tail: format!(
                "SELECT MAX(sequence), COUNT(*) FROM infer_response_events \
                 WHERE response_id = {}",
                Parameters::new(spell).next()
            ),
            append_event: format!(
                "INSERT INTO infer_response_events (response_id, sequence, payload) \
                 VALUES ({})",
                append.list(APPEND_PARAMETERS)
            ),
            events: format!(
                "SELECT e.sequence, e.payload FROM infer_response_events e \
                 JOIN infer_responses r ON r.id = e.response_id \
                 WHERE e.response_id = {} AND e.sequence > {} AND NOT ({}) \
                 ORDER BY e.sequence",
                events.next(),
                events.next(),
                expired("r.", events.next())
            ),
            delete: format!(
                "DELETE FROM infer_responses WHERE id = {}",
                Parameters::new(spell).next()
            ),
        }
    }
}

fn terminal_list() -> String {
    TERMINAL
        .iter()
        .map(|status| format!("'{}'", status.as_str()))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The tables, created when missing. `integer` and `bytes` are the backend's 64-bit integer and
/// byte-string column types.
pub fn schema(integer: &str, bytes: &str) -> String {
    let terminal = terminal_list();
    format!(
        "CREATE TABLE IF NOT EXISTS infer_response_store_meta (
            key TEXT PRIMARY KEY,
            value {integer} NOT NULL
        );
        CREATE TABLE IF NOT EXISTS infer_responses (
            id TEXT PRIMARY KEY,
            request_id {integer} NOT NULL,
            model TEXT NOT NULL,
            created_at {integer} NOT NULL,
            completed_at {integer},
            status TEXT NOT NULL,
            incomplete TEXT,
            error_code TEXT,
            error_message TEXT,
            input_tokens {integer},
            output_tokens {integer},
            reasoning_tokens {integer},
            cached_tokens {integer},
            cache_write_tokens {integer},
            previous_response_id TEXT,
            request_input {bytes} NOT NULL,
            response {bytes} NOT NULL,
            payload_version {integer} NOT NULL,
            expires_at {integer}
        );
        CREATE INDEX IF NOT EXISTS infer_responses_expiry ON infer_responses (expires_at)
            WHERE status IN ({terminal});
        CREATE TABLE IF NOT EXISTS infer_response_events (
            response_id TEXT NOT NULL REFERENCES infer_responses (id) ON DELETE CASCADE,
            sequence {integer} NOT NULL,
            payload {bytes} NOT NULL,
            PRIMARY KEY (response_id, sequence)
        );"
    )
}
