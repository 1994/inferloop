//! Tables, migrations and statements, spelled once per dialect when a store opens.
use crate::database::Dialect;

/// Migrations in order; the database records the last one applied.
pub fn migrations(dialect: &Dialect) -> Vec<String> {
    vec![initial(dialect)]
}

fn initial(dialect: &Dialect) -> String {
    let Dialect {
        integer,
        bytes,
        key,
        version_source,
        ..
    } = dialect;
    format!(
        "{version_source};
        CREATE TABLE IF NOT EXISTS infer_records (
            namespace {key} NOT NULL,
            id {key} NOT NULL,
            version {integer} NOT NULL,
            state {integer} NOT NULL,
            state_bit {integer} NOT NULL,
            disposable {integer} NOT NULL,
            expires_at {integer},
            value {bytes} NOT NULL,
            PRIMARY KEY (namespace, id)
        );
        CREATE INDEX IF NOT EXISTS infer_records_expiry ON infer_records (expires_at);
        CREATE TABLE IF NOT EXISTS infer_record_logs (
            namespace {key} NOT NULL,
            id {key} NOT NULL,
            sequence {integer} NOT NULL,
            value {bytes} NOT NULL,
            PRIMARY KEY (namespace, id, sequence),
            FOREIGN KEY (namespace, id) REFERENCES infer_records (namespace, id) ON DELETE CASCADE
        )"
    )
}

/// The table that records the schema version; created before any migration runs.
pub fn meta_table(dialect: &Dialect) -> String {
    format!(
        "CREATE TABLE IF NOT EXISTS infer_storage_meta (key {} PRIMARY KEY, value {} NOT NULL)",
        dialect.key, dialect.integer
    )
}

/// Spells the bound parameters of one statement in order.
struct Parameters {
    spell: fn(usize) -> String,
    next: usize,
}

impl Parameters {
    const fn new(spell: fn(usize) -> String) -> Self {
        Self { spell, next: 1 }
    }

    fn next(&mut self) -> String {
        let spelled = (self.spell)(self.next);
        self.next += 1;
        spelled
    }
}

/// Every statement a store runs, in the dialect's syntax.
#[derive(Debug)]
pub struct Statements {
    pub schema_version: String,
    pub set_schema_version: String,
    pub next_version: &'static str,
    pub purge: String,
    pub purge_key: String,
    pub get: String,
    pub current: String,
    pub insert_absent: String,
    pub upsert: String,
    pub upsert_unless_state: String,
    pub update_version: String,
    pub delete: String,
    pub log_tail: String,
    pub append: String,
    pub log: String,
}

/// Columns a write binds, in this order, after the key.
const WRITE_COLUMNS: [&str; 6] = [
    "version",
    "state",
    "state_bit",
    "disposable",
    "expires_at",
    "value",
];

/// `table`'s record has expired at the time bound to `now`.
fn expired(table: &str, now: &str) -> String {
    format!(
        "{table}disposable = 1 AND {table}expires_at IS NOT NULL AND {table}expires_at <= {now}"
    )
}

/// `table`'s key equals the next two parameters.
fn key(parameters: &mut Parameters, table: &str) -> String {
    format!(
        "{table}namespace = {} AND {table}id = {}",
        parameters.next(),
        parameters.next()
    )
}

/// An insert binding the key, then [`WRITE_COLUMNS`].
fn insert(parameters: &mut Parameters) -> String {
    let names: Vec<&str> = ["namespace", "id"]
        .iter()
        .chain(&WRITE_COLUMNS)
        .copied()
        .collect();
    let values: Vec<String> = names.iter().map(|_| parameters.next()).collect();
    format!(
        "INSERT INTO infer_records ({}) VALUES ({})",
        names.join(", "),
        values.join(", ")
    )
}

/// Upserts binding the key and [`WRITE_COLUMNS`]: unconditional, and unless the current
/// state's bit is in the set bound last.
fn upserts(spell: fn(usize) -> String) -> (String, String) {
    let mut parameters = Parameters::new(spell);
    let updates = WRITE_COLUMNS
        .iter()
        .map(|column| format!("{column} = excluded.{column}"))
        .collect::<Vec<_>>()
        .join(", ");
    let upsert = format!(
        "{} ON CONFLICT (namespace, id) DO UPDATE SET {updates}",
        insert(&mut parameters)
    );
    let unless_state = format!(
        "{upsert} WHERE ({} & infer_records.state_bit) = 0",
        parameters.next()
    );
    (upsert, unless_state)
}

/// An update binding [`WRITE_COLUMNS`], the key, then the version it expects.
fn update_version(spell: fn(usize) -> String) -> String {
    let mut parameters = Parameters::new(spell);
    let columns = WRITE_COLUMNS
        .iter()
        .map(|column| format!("{column} = {}", parameters.next()))
        .collect::<Vec<_>>()
        .join(", ");
    let key = key(&mut parameters, "");
    format!(
        "UPDATE infer_records SET {columns} WHERE {key} AND version = {}",
        parameters.next()
    )
}

fn log(spell: fn(usize) -> String) -> String {
    let mut parameters = Parameters::new(spell);
    format!(
        "SELECT l.sequence, l.value FROM infer_record_logs l \
         JOIN infer_records r ON r.namespace = l.namespace AND r.id = l.id \
         WHERE {} AND l.sequence > {} AND NOT ({}) ORDER BY l.sequence",
        key(&mut parameters, "l."),
        parameters.next(),
        expired("r.", &parameters.next())
    )
}

impl Statements {
    pub fn new(dialect: &Dialect) -> Self {
        let spell = dialect.placeholder;
        let fresh = || Parameters::new(spell);
        let (upsert, upsert_unless_state) = upserts(spell);
        let mut get = fresh();
        let mut purge_key = fresh();
        let mut set_version = fresh();
        let mut append = fresh();
        Self {
            schema_version: format!(
                "SELECT value FROM infer_storage_meta WHERE key = {}",
                fresh().next()
            ),
            set_schema_version: format!(
                "INSERT INTO infer_storage_meta (key, value) VALUES ({}, {}) \
                 ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                set_version.next(),
                set_version.next()
            ),
            next_version: dialect.next_version,
            purge: format!(
                "DELETE FROM infer_records WHERE {}",
                expired("", &fresh().next())
            ),
            purge_key: format!(
                "DELETE FROM infer_records WHERE {} AND {}",
                key(&mut purge_key, ""),
                expired("", &purge_key.next())
            ),
            get: format!(
                "SELECT version, state, disposable, expires_at, value FROM infer_records \
                 WHERE {} AND NOT ({})",
                key(&mut get, ""),
                expired("", &get.next())
            ),
            current: format!(
                "SELECT version, state FROM infer_records WHERE {}{}",
                key(&mut fresh(), ""),
                dialect.row_lock
            ),
            insert_absent: format!(
                "{} ON CONFLICT (namespace, id) DO NOTHING",
                insert(&mut fresh())
            ),
            upsert,
            upsert_unless_state,
            update_version: update_version(spell),
            delete: format!("DELETE FROM infer_records WHERE {}", key(&mut fresh(), "")),
            log_tail: format!(
                "SELECT MAX(sequence), COUNT(*) FROM infer_record_logs WHERE {}",
                key(&mut fresh(), "")
            ),
            append: format!(
                "INSERT INTO infer_record_logs (namespace, id, sequence, value) VALUES ({}, {}, {}, {})",
                append.next(),
                append.next(),
                append.next(),
                append.next()
            ),
            log: log(spell),
        }
    }
}
