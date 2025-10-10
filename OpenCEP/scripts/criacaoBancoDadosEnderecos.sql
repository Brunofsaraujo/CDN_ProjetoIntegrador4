PRAGMA foreign_keys = ON;

CREATE TABLE uf (
    cod_uf INTEGER PRIMARY KEY,
    nome TEXT,
    sigla TEXT
);

CREATE TABLE municipio (
    cod_municipio INTEGER PRIMARY KEY,
    cod_uf INTEGER NOT NULL,
    nome TEXT,
    FOREIGN KEY (cod_uf) REFERENCES uf(cod_uf) ON DELETE RESTRICT ON UPDATE CASCADE
);

CREATE TABLE endereco (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cod_unico_endereco INTEGER NOT NULL,
    cod_uf INTEGER,
    cod_municipio INTEGER,
    cod_distrito INTEGER,
    cod_subdistrito INTEGER,
    cod_setor TEXT,
    num_quadra INTEGER,
    num_face INTEGER,
    cep TEXT,
    dsc_localidade TEXT,
    nom_tipo_seglogr TEXT,
    nom_titulo_seglogr TEXT,
    nom_seglogr TEXT,
    num_endereco INTEGER,
    dsc_modificador TEXT,
    nom_comp_elem1 TEXT,
    val_comp_elem1 TEXT,
    nom_comp_elem2 TEXT,
    val_comp_elem2 TEXT,
    nom_comp_elem3 TEXT,
    val_comp_elem3 TEXT,
    nom_comp_elem4 TEXT,
    val_comp_elem4 TEXT,
    nom_comp_elem5 TEXT,
    val_comp_elem5 TEXT,
    latitude REAL,
    longitude REAL,
    nv_geo_coord INTEGER,
    cod_especie INTEGER NOT NULL DEFAULT 0,
    dsc_estabelecimento TEXT,
    cod_indicador_estab_endereco INTEGER,
    cod_indicador_const_endereco INTEGER,
    cod_indicador_finalidade_const INTEGER,
    cod_tipo_especi INTEGER NOT NULL DEFAULT 0,
    created_at DATETIME DEFAULT (datetime('now')),
    updated_at DATETIME DEFAULT (datetime('now')),
    UNIQUE(cod_unico_endereco, cod_especie, cod_tipo_especi)
);

CREATE INDEX IF NOT EXISTS idx_endereco_cep ON endereco(cep);

CREATE INDEX IF NOT EXISTS idx_endereco_nom_seglogr ON endereco(nom_seglogr);

CREATE INDEX IF NOT EXISTS idx_endereco_dsc_localidade ON endereco(dsc_localidade);

CREATE INDEX IF NOT EXISTS idx_endereco_cod_uf ON endereco(cod_uf);

CREATE TABLE endereco_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    endereco_id INTEGER NOT NULL,
    version_ts DATETIME DEFAULT (datetime('now')),
    change_type TEXT NOT NULL,
    -- 'INSERT', 'UPDATE', 'DELETE'
    changed_by TEXT,
    cod_unico_endereco INTEGER,
    cod_uf INTEGER,
    cod_municipio INTEGER,
    cod_distrito INTEGER,
    cod_subdistrito INTEGER,
    cod_setor INTEGER,
    num_quadra INTEGER,
    num_face INTEGER,
    cep TEXT,
    dsc_localidade TEXT,
    nom_tipo_seglogr TEXT,
    nom_titulo_seglogr TEXT,
    nom_seglogr TEXT,
    num_endereco INTEGER,
    dsc_modificador TEXT,
    nom_comp_elem1 TEXT,
    val_comp_elem1 TEXT,
    nom_comp_elem2 TEXT,
    val_comp_elem2 TEXT,
    nom_comp_elem3 TEXT,
    val_comp_elem3 TEXT,
    nom_comp_elem4 TEXT,
    val_comp_elem4 TEXT,
    nom_comp_elem5 TEXT,
    val_comp_elem5 TEXT,
    latitude REAL,
    longitude REAL,
    nv_geo_coord INTEGER,
    cod_especie INTEGER,
    dsc_estabelecimento TEXT,
    cod_indicador_estab_endereco INTEGER,
    cod_indicador_const_endereco INTEGER,
    cod_indicador_finalidade_const INTEGER,
    cod_tipo_especi TEXT,
    snapshot_created_at DATETIME,
    snapshot_updated_at DATETIME
);

-- Trigger: on INSERT
CREATE TRIGGER trg_endereco_after_insert
AFTER
INSERT
    ON endereco BEGIN
INSERT INTO
    endereco_history (
        endereco_id,
        change_type,
        cod_unico_endereco,
        cod_uf,
        cod_municipio,
        cod_distrito,
        cod_subdistrito,
        cod_setor,
        num_quadra,
        num_face,
        cep,
        dsc_localidade,
        nom_tipo_seglogr,
        nom_titulo_seglogr,
        nom_seglogr,
        num_endereco,
        dsc_modificador,
        nom_comp_elem1,
        val_comp_elem1,
        nom_comp_elem2,
        val_comp_elem2,
        nom_comp_elem3,
        val_comp_elem3,
        nom_comp_elem4,
        val_comp_elem4,
        nom_comp_elem5,
        val_comp_elem5,
        latitude,
        longitude,
        nv_geo_coord,
        cod_especie,
        dsc_estabelecimento,
        cod_indicador_estab_endereco,
        cod_indicador_const_endereco,
        cod_indicador_finalidade_const,
        cod_tipo_especi,
        snapshot_created_at,
        snapshot_updated_at
    )
VALUES
    (
        NEW.id,
        'INSERT',
        NEW.cod_unico_endereco,
        NEW.cod_uf,
        NEW.cod_municipio,
        NEW.cod_distrito,
        NEW.cod_subdistrito,
        NEW.cod_setor,
        NEW.num_quadra,
        NEW.num_face,
        NEW.cep,
        NEW.dsc_localidade,
        NEW.nom_tipo_seglogr,
        NEW.nom_titulo_seglogr,
        NEW.nom_seglogr,
        NEW.num_endereco,
        NEW.dsc_modificador,
        NEW.nom_comp_elem1,
        NEW.val_comp_elem1,
        NEW.nom_comp_elem2,
        NEW.val_comp_elem2,
        NEW.nom_comp_elem3,
        NEW.val_comp_elem3,
        NEW.nom_comp_elem4,
        NEW.val_comp_elem4,
        NEW.nom_comp_elem5,
        NEW.val_comp_elem5,
        NEW.latitude,
        NEW.longitude,
        NEW.nv_geo_coord,
        NEW.cod_especie,
        NEW.dsc_estabelecimento,
        NEW.cod_indicador_estab_endereco,
        NEW.cod_indicador_const_endereco,
        NEW.cod_indicador_finalidade_const,
        NEW.cod_tipo_especi,
        NEW.created_at,
        NEW.updated_at
    );

END;

-- Trigger: before UPDATE (atualiza timestamp)
CREATE TRIGGER trg_endereco_before_update BEFORE
UPDATE
    ON endereco BEGIN
UPDATE
    endereco
SET
    updated_at = datetime('now')
WHERE
    id = OLD.id;

END;

-- Trigger: after UPDATE (salva histórico)
CREATE TRIGGER trg_endereco_after_update
AFTER
UPDATE
    ON endereco BEGIN
INSERT INTO
    endereco_history (
        endereco_id,
        change_type,
        cod_unico_endereco,
        cod_uf,
        cod_municipio,
        cod_distrito,
        cod_subdistrito,
        cod_setor,
        num_quadra,
        num_face,
        cep,
        dsc_localidade,
        nom_tipo_seglogr,
        nom_titulo_seglogr,
        nom_seglogr,
        num_endereco,
        dsc_modificador,
        nom_comp_elem1,
        val_comp_elem1,
        nom_comp_elem2,
        val_comp_elem2,
        nom_comp_elem3,
        val_comp_elem3,
        nom_comp_elem4,
        val_comp_elem4,
        nom_comp_elem5,
        val_comp_elem5,
        latitude,
        longitude,
        nv_geo_coord,
        cod_especie,
        dsc_estabelecimento,
        cod_indicador_estab_endereco,
        cod_indicador_const_endereco,
        cod_indicador_finalidade_const,
        cod_tipo_especi,
        snapshot_created_at,
        snapshot_updated_at
    )
VALUES
    (
        OLD.id,
        'UPDATE',
        OLD.cod_unico_endereco,
        OLD.cod_uf,
        OLD.cod_municipio,
        OLD.cod_distrito,
        OLD.cod_subdistrito,
        OLD.cod_setor,
        OLD.num_quadra,
        OLD.num_face,
        OLD.cep,
        OLD.dsc_localidade,
        OLD.nom_tipo_seglogr,
        OLD.nom_titulo_seglogr,
        OLD.nom_seglogr,
        OLD.num_endereco,
        OLD.dsc_modificador,
        OLD.nom_comp_elem1,
        OLD.val_comp_elem1,
        OLD.nom_comp_elem2,
        OLD.val_comp_elem2,
        OLD.nom_comp_elem3,
        OLD.val_comp_elem3,
        OLD.nom_comp_elem4,
        OLD.val_comp_elem4,
        OLD.nom_comp_elem5,
        OLD.val_comp_elem5,
        OLD.latitude,
        OLD.longitude,
        OLD.nv_geo_coord,
        OLD.cod_especie,
        OLD.dsc_estabelecimento,
        OLD.cod_indicador_estab_endereco,
        OLD.cod_indicador_const_endereco,
        OLD.cod_indicador_finalidade_const,
        OLD.cod_tipo_especi,
        OLD.created_at,
        OLD.updated_at
    );

END;

-- Trigger: on DELETE
CREATE TRIGGER trg_endereco_before_delete BEFORE DELETE ON endereco BEGIN
INSERT INTO
    endereco_history (
        endereco_id,
        change_type,
        cod_unico_endereco,
        cod_uf,
        cod_municipio,
        cod_distrito,
        cod_subdistrito,
        cod_setor,
        num_quadra,
        num_face,
        cep,
        dsc_localidade,
        nom_tipo_seglogr,
        nom_titulo_seglogr,
        nom_seglogr,
        num_endereco,
        dsc_modificador,
        nom_comp_elem1,
        val_comp_elem1,
        nom_comp_elem2,
        val_comp_elem2,
        nom_comp_elem3,
        val_comp_elem3,
        nom_comp_elem4,
        val_comp_elem4,
        nom_comp_elem5,
        val_comp_elem5,
        latitude,
        longitude,
        nv_geo_coord,
        cod_especie,
        dsc_estabelecimento,
        cod_indicador_estab_endereco,
        cod_indicador_const_endereco,
        cod_indicador_finalidade_const,
        cod_tipo_especi,
        snapshot_created_at,
        snapshot_updated_at
    )
VALUES
    (
        OLD.id,
        'DELETE',
        OLD.cod_unico_endereco,
        OLD.cod_uf,
        OLD.cod_municipio,
        OLD.cod_distrito,
        OLD.cod_subdistrito,
        OLD.cod_setor,
        OLD.num_quadra,
        OLD.num_face,
        OLD.cep,
        OLD.dsc_localidade,
        OLD.nom_tipo_seglogr,
        OLD.nom_titulo_seglogr,
        OLD.nom_seglogr,
        OLD.num_endereco,
        OLD.dsc_modificador,
        OLD.nom_comp_elem1,
        OLD.val_comp_elem1,
        OLD.nom_comp_elem2,
        OLD.val_comp_elem2,
        OLD.nom_comp_elem3,
        OLD.val_comp_elem3,
        OLD.nom_comp_elem4,
        OLD.val_comp_elem4,
        OLD.nom_comp_elem5,
        OLD.val_comp_elem5,
        OLD.latitude,
        OLD.longitude,
        OLD.nv_geo_coord,
        OLD.cod_especie,
        OLD.dsc_estabelecimento,
        OLD.cod_indicador_estab_endereco,
        OLD.cod_indicador_const_endereco,
        OLD.cod_indicador_finalidade_const,
        OLD.cod_tipo_especi,
        OLD.created_at,
        OLD.updated_at
    );

END;

CREATE VIEW vw_endereco_full AS
SELECT
    e.*,
    m.nome AS municipio_nome,
    u.nome AS uf_nome,
    u.sigla AS uf_sigla,
    t.nome AS tipo_logradouro_nome
FROM
    endereco e
    LEFT JOIN municipio m ON e.cod_municipio = m.cod_municipio
    LEFT JOIN uf u ON e.cod_uf = u.cod_uf
    LEFT JOIN tipo_logradouro t ON e.nom_tipo_seglogr = t.nome;

-- Example queries:
-- 1) Buscar endereços por CEP
-- SELECT * FROM vw_endereco_full WHERE cep = '01001-000';
-- 2) Histórico de versões de um endereço (por cod_unico_endereco)
-- SELECT h.* FROM endereco_history h
-- JOIN endereco e ON e.id = h.endereco_id
-- WHERE e.cod_unico_endereco = 123456
-- ORDER BY h.version_ts DESC;
-- 3) Endereços de um município
-- SELECT * FROM vw_endereco_full WHERE cod_municipio = 3550308;
-- 4) Consulta de proximidade (bounding box)
-- SELECT *, ((latitude - :lat)*(latitude - :lat) + (longitude - :lon)*(longitude - :lon)) AS dist2
-- FROM endereco
-- WHERE latitude BETWEEN :lat - :delta AND :lat + :delta
--   AND longitude BETWEEN :lon - :delta AND :lon + :delta
-- ORDER BY dist2 LIMIT 100;
-- Notes:
-- - As triggers populam endereco_history em INSERT/UPDATE/DELETE, contendo o snapshot dos dados
-- - Recomenda-se popular as tabelas de referência (uf, municipio, tipo_logradouro, especie) a partir das tabelas oficiais do IBGE
-- - Para consultas geoespaciais mais precisas, avaliar uso da extensão Spatialite ou criação de UDFs (ex.: função distance(lat1, lon1, lat2, lon2))