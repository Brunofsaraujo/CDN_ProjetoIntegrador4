import os
import zipfile
import sqlite3
import pandas as pd
import logging
from flask import request, jsonify
from werkzeug.utils import secure_filename
from flask_openapi3 import OpenAPI, Info
from typing import List, Dict, Tuple, Optional
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

LOG_FILE = os.path.join(os.path.dirname(__file__), "app.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

info = Info(
    title="OpenCEP API",
    version="1.0.0",
    description="API pública para importação e consulta de CEPs",
)
app = OpenAPI(__name__, info=info)

CORS(app, resources={r"/api/*": {"origins": "*"}})

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["1000 per day", "200 per hour"],
    storage_uri="memory://",
    strategy="fixed-window",
)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
DB_PATH = "OpenCEP.db"
MAX_CONTENT_LENGTH = 2048 * 1024 * 1024  # 2GB

app.config.update(UPLOAD_FOLDER=UPLOAD_FOLDER, MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

DTYPE_MAP = {
    "cod_unico_endereco": "Int64",
    "cod_uf": "Int64",
    "cod_municipio": "Int64",
    "cod_distrito": "Int64",
    "cod_subdistrito": "Int64",
    "cod_setor": "Int64",
    "num_quadra": "Int64",
    "num_face": "Int64",
    "cep": "string",
    "dsc_localidade": "string",
    "nom_tipo_seglogr": "string",
    "nom_titulo_seglogr": "string",
    "nom_seglogr": "string",
    "num_endereco": "Int64",
    "dsc_modificador": "string",
    "nom_comp_elem1": "string",
    "val_comp_elem1": "string",
    "nom_comp_elem2": "string",
    "val_comp_elem2": "string",
    "nom_comp_elem3": "string",
    "val_comp_elem3": "string",
    "nom_comp_elem4": "string",
    "val_comp_elem4": "string",
    "nom_comp_elem5": "string",
    "val_comp_elem5": "string",
    "latitude": "float64",
    "longitude": "float64",
    "nv_geo_coord": "Int64",
    "cod_especie": "Int64",
    "dsc_estabelecimento": "string",
    "cod_indicador_estab_endereco": "Int64",
    "cod_indicador_const_endereco": "Int64",
    "cod_indicador_finalidade_const": "Int64",
    "cod_tipo_especi": "Int64",
}

KEY_COLUMNS = ["cod_unico_endereco", "cod_especie", "cod_tipo_especi"]

UF_INFO = {
    11: {"sigla": "RO", "estado": "Rondônia", "regiao": "Norte"},
    12: {"sigla": "AC", "estado": "Acre", "regiao": "Norte"},
    13: {"sigla": "AM", "estado": "Amazonas", "regiao": "Norte"},
    14: {"sigla": "RR", "estado": "Roraima", "regiao": "Norte"},
    15: {"sigla": "PA", "estado": "Pará", "regiao": "Norte"},
    16: {"sigla": "AP", "estado": "Amapá", "regiao": "Norte"},
    17: {"sigla": "TO", "estado": "Tocantins", "regiao": "Norte"},
    21: {"sigla": "MA", "estado": "Maranhão", "regiao": "Nordeste"},
    22: {"sigla": "PI", "estado": "Piauí", "regiao": "Nordeste"},
    23: {"sigla": "CE", "estado": "Ceará", "regiao": "Nordeste"},
    24: {"sigla": "RN", "estado": "Rio Grande do Norte", "regiao": "Nordeste"},
    25: {"sigla": "PB", "estado": "Paraíba", "regiao": "Nordeste"},
    26: {"sigla": "PE", "estado": "Pernambuco", "regiao": "Nordeste"},
    27: {"sigla": "AL", "estado": "Alagoas", "regiao": "Nordeste"},
    28: {"sigla": "SE", "estado": "Sergipe", "regiao": "Nordeste"},
    29: {"sigla": "BA", "estado": "Bahia", "regiao": "Nordeste"},
    31: {"sigla": "MG", "estado": "Minas Gerais", "regiao": "Sudeste"},
    32: {"sigla": "ES", "estado": "Espírito Santo", "regiao": "Sudeste"},
    33: {"sigla": "RJ", "estado": "Rio de Janeiro", "regiao": "Sudeste"},
    35: {"sigla": "SP", "estado": "São Paulo", "regiao": "Sudeste"},
    41: {"sigla": "PR", "estado": "Paraná", "regiao": "Sul"},
    42: {"sigla": "SC", "estado": "Santa Catarina", "regiao": "Sul"},
    43: {"sigla": "RS", "estado": "Rio Grande do Sul", "regiao": "Sul"},
    50: {"sigla": "MS", "estado": "Mato Grosso do Sul", "regiao": "Centro-Oeste"},
    51: {"sigla": "MT", "estado": "Mato Grosso", "regiao": "Centro-Oeste"},
    52: {"sigla": "GO", "estado": "Goiás", "regiao": "Centro-Oeste"},
    53: {"sigla": "DF", "estado": "Distrito Federal", "regiao": "Centro-Oeste"},
}

# ========== CAMADA DE EXTRAÇÃO (Extract) ==========


class DataExtractor:
    @staticmethod
    def validate_file(file) -> Tuple[bool, str]:
        if not file or file.filename == "":
            return False, "Nenhum arquivo selecionado"
        if not file.filename.endswith(".zip"):
            return False, "Formato inválido. Apenas arquivos .zip são aceitos"
        return True, "Arquivo válido"

    @staticmethod
    def extract_zip(file_path: str, destination: str) -> List[str]:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            csv_files = [f for f in zip_ref.namelist() if f.endswith((".csv", ".txt"))]
            zip_ref.extractall(destination)
        logger.info(f"Extraídos {len(csv_files)} arquivos CSV/TXT do ZIP")
        return csv_files

    @staticmethod
    def save_uploaded_file(file, folder: str) -> str:
        filename = secure_filename(file.filename)
        filepath = os.path.join(folder, filename)
        file.save(filepath)
        logger.info(f"Arquivo salvo em: {filepath}")
        return filepath


# ========== CAMADA DE TRANSFORMAÇÃO (Transform) ==========


class DataTransformer:
    @staticmethod
    def read_csv(filepath: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(
                filepath,
                sep=";",
                encoding="utf-8",
                on_bad_lines="skip",
                dtype=str,
                keep_default_na=False,
            )
            df.columns = df.columns.str.lower()
            logger.info(f"CSV lido: {filepath} - {len(df)} linhas")
            return df
        except Exception as e:
            logger.error(f"Erro ao ler CSV {filepath}: {str(e)}")
            raise

    @staticmethod
    def apply_schema(df: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
        for col, dtype in schema.items():
            if col in df.columns:
                try:
                    if dtype == "string":
                        df[col] = (
                            df[col]
                            .astype(str)
                            .str.strip()
                            .replace(["nan", "NaN", "NULL", ""], None)
                        )
                    else:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    df[col] = df[col].astype(dtype)
                except Exception as e:
                    logger.warning(f"Erro ao converter coluna {col}: {str(e)}")
        return df

    @staticmethod
    def normalize_keys(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
        for col in key_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype("Int64")
        return df

    @staticmethod
    def validate_data(df: pd.DataFrame) -> Tuple[bool, str, pd.DataFrame]:
        if df.empty:
            return False, "DataFrame vazio", df
        df = df.dropna(how="all")
        key_check = df[KEY_COLUMNS].notna().any(axis=1)
        invalid_rows = (~key_check).sum()
        if invalid_rows > 0:
            logger.warning(
                f"{invalid_rows} linhas sem chaves primárias serão removidas"
            )
            df = df[key_check]
        if df.empty:
            return False, "Nenhum registro válido encontrado", df
        logger.info(f"Validação OK: {len(df)} registros válidos")
        return True, "Dados válidos", df


# ========== CAMADA DE CARGA (Load) ==========


class DataLoader:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def upsert_dataframe(self, df: pd.DataFrame, key_cols: List[str]) -> int:
        if df.empty:
            return 0
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            columns, placeholders = list(df.columns), ", ".join(["?"] * len(df.columns))
            update_cols = [col for col in columns if col not in key_cols]
            update_sets = ", ".join([f"{col} = excluded.{col}" for col in update_cols])
            query = f"INSERT INTO endereco ({', '.join(columns)}) VALUES ({placeholders}) ON CONFLICT({', '.join(key_cols)}) DO UPDATE SET {update_sets}"
            data = [
                tuple(
                    (
                        None
                        if pd.isna(val)
                        else (0 if col in key_cols and pd.isna(val) else val)
                    )
                    for col, val in row.items()
                )
                for _, row in df.iterrows()
            ]
            cursor.executemany(query, data)
            affected_rows = cursor.rowcount
            conn.commit()
            logger.info(f"UPSERT executado: {affected_rows} linhas afetadas")
            return affected_rows


# ========== CAMADA DE CONSULTA (Query) ==========


class DataQuery:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_statistics(self) -> Dict:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*), COUNT(DISTINCT cep), COUNT(DISTINCT dsc_localidade), COUNT(DISTINCT cod_uf) FROM endereco"
                )
                total, ceps, cities, states = cursor.fetchone()
                return {
                    "total_records": total,
                    "unique_ceps": ceps,
                    "unique_cities": cities,
                    "unique_states": states,
                }
        except Exception as e:
            logger.error(f"Erro ao obter estatísticas: {str(e)}")
            return {
                "total_records": 0,
                "unique_ceps": 0,
                "unique_cities": 0,
                "unique_states": 0,
            }

    def get_states_list(self) -> List[Dict]:
        try:
            states_list = [{"cod_uf": code, **info} for code, info in UF_INFO.items()]
            return sorted(states_list, key=lambda x: x["cod_uf"])
        except Exception as e:
            logger.error(f"Erro ao gerar lista de estados: {str(e)}")
            return []

    def _format_results(self, rows: List[sqlite3.Row]) -> List[Dict]:
        results = []
        for row in rows:
            row_dict = dict(row)
            uf_info = UF_INFO.get(row_dict.get("cod_uf"), {})
            logradouro = f"{row_dict.get('nom_tipo_seglogr') or ''} {row_dict.get('nom_seglogr') or ''}".strip()
            complemento = f"{row_dict.get('nom_comp_elem1') or ''} {row_dict.get('val_comp_elem1') or ''} {row_dict.get('nom_comp_elem2') or ''} {row_dict.get('val_comp_elem2') or ''}".strip()
            results.append(
                {
                    "cep": row_dict.get("cep"),
                    "logradouro": logradouro,
                    "numero": row_dict.get("num_endereco"),
                    "bairro": row_dict.get("dsc_localidade"),
                    "cidade": row_dict.get("nome"),
                    "complemento": complemento,
                    "uf": uf_info.get("sigla"),
                    "estado": uf_info.get("estado"),
                    "regiao": uf_info.get("regiao"),
                    "ibge": (
                        str(row_dict.get("cod_municipio"))
                        if row_dict.get("cod_municipio")
                        else ""
                    ),
                }
            )
        return results

    def search_addresses(
        self,
        cep: Optional[str] = None,
        logradouro: Optional[str] = None,
        localidade: Optional[str] = None,
        cod_uf: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
        group_by_street: bool = False,
    ) -> Tuple[List[Dict], int]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                where_clauses, params = [], []
                if cep:
                    where_clauses.append("cep LIKE ?")
                    params.append(f"%{cep}%")
                if logradouro:
                    where_clauses.append(
                        "(nom_tipo_seglogr || ' ' || nom_seglogr) LIKE ?"
                    )
                    params.append(f"%{logradouro}%")
                if localidade:
                    where_clauses.append("M.nome LIKE ? COLLATE NOCASE")
                    params.append(f"%{localidade}%")
                if cod_uf:
                    where_clauses.append("M.cod_uf = ?")
                    params.append(cod_uf)
                where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

                base_query = f"FROM endereco E JOIN municipio M ON E.cod_municipio = M.cod_municipio AND E.cod_uf = M.cod_uf WHERE {where_sql}"

                if group_by_street:
                    if cep and not any([logradouro, localidade, cod_uf]):
                        query = f"SELECT * {base_query} ORDER BY CASE WHEN nom_tipo_seglogr LIKE 'RUA%' THEN 1 WHEN nom_tipo_seglogr LIKE 'AV%' THEN 2 ELSE 3 END, nom_seglogr ASC LIMIT 1"
                        cursor.execute(query, params)
                        rows = cursor.fetchall()
                        return self._format_results(rows), len(rows)

                    count_query = f"SELECT COUNT(*) FROM (SELECT 1 {base_query} GROUP BY cep, nom_tipo_seglogr, nom_seglogr)"
                    query = f"SELECT MIN(E.rowid) as rowid, E.*, UPPER(M.nome) as nome {base_query} GROUP BY cep, nom_tipo_seglogr, nom_seglogr ORDER BY cep, nom_seglogr LIMIT ? OFFSET ?"
                else:
                    count_query = f"SELECT COUNT(*) {base_query}"
                    query = f"SELECT E.*, UPPER(M.nome) as nome {base_query} ORDER BY cep, nom_seglogr LIMIT ? OFFSET ?"

                cursor.execute(count_query, params)
                total_count = cursor.fetchone()[0]
                cursor.execute(query, params + [limit, offset])
                rows = cursor.fetchall()
                return self._format_results(rows), total_count
        except Exception as e:
            logger.error(f"Erro na busca: {str(e)}")
            return [], 0

    def bulk_search_addresses(
        self,
        filters: List[Dict],
        limit: int = 100,
        offset: int = 0,
        group_by_street: bool = False,
    ) -> Tuple[List[Dict], int]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                bulk_where, params, uf_map = (
                    [],
                    [],
                    {info["sigla"].upper(): code for code, info in UF_INFO.items()},
                )
                for f in filters:
                    single_filter = []
                    if f.get("cep"):
                        single_filter.append("cep LIKE ?")
                        params.append(f"%{f['cep'].replace('-', '')}%")
                    if f.get("logradouro"):
                        single_filter.append(
                            "(nom_tipo_seglogr || ' ' || nom_seglogr) LIKE ?"
                        )
                        params.append(f"%{f['logradouro']}%")
                    if f.get("localidade"):
                        single_filter.append("M.nome LIKE ? COLLATE NOCASE")
                        params.append(f"%{f['localidade']}%")
                    if f.get("uf") and f["uf"].upper() in uf_map:
                        single_filter.append("M.cod_uf = ?")
                        params.append(uf_map[f["uf"].upper()])
                    if single_filter:
                        bulk_where.append(f"({' AND '.join(single_filter)})")
                if not bulk_where:
                    return [], 0
                where_sql = " OR ".join(bulk_where)

                if group_by_street:
                    ranked_query = f"""
                        WITH RankedAddresses AS (
                            SELECT E.*, UPPER(M.nome) as nome, ROW_NUMBER() OVER (PARTITION BY E.cep ORDER BY CASE WHEN E.nom_tipo_seglogr LIKE 'RUA%' THEN 1 WHEN E.nom_tipo_seglogr LIKE 'AV%' THEN 2 ELSE 3 END, E.nom_seglogr ASC) as rn
                            FROM endereco E JOIN municipio M ON E.cod_municipio = M.cod_municipio AND E.cod_uf = M.cod_uf WHERE {where_sql}
                        ) SELECT * FROM RankedAddresses WHERE rn = 1
                    """
                    count_query = f"SELECT COUNT(*) FROM ({ranked_query})"
                    query = f"{ranked_query} ORDER BY cep, nom_seglogr LIMIT ? OFFSET ?"
                else:
                    base_query = f"FROM endereco E JOIN municipio M ON E.cod_municipio = M.cod_municipio AND E.cod_uf = M.cod_uf WHERE {where_sql}"
                    count_query = f"SELECT COUNT(*) {base_query}"
                    query = f"SELECT E.*, UPPER(M.nome) as nome {base_query} ORDER BY cep, nom_seglogr LIMIT ? OFFSET ?"

                cursor.execute(count_query, params)
                total_count = cursor.fetchone()[0]
                cursor.execute(query, params + [limit, offset])
                rows = cursor.fetchall()
                return self._format_results(rows), total_count
        except Exception as e:
            logger.error(f"Erro na busca em lote: {str(e)}")
            return [], 0


# ========== ORQUESTRADOR ETL ==========


class ETLPipeline:
    def __init__(self, upload_folder: str, db_path: str):
        self.extractor, self.transformer, self.loader = (
            DataExtractor(),
            DataTransformer(),
            DataLoader(db_path),
        )
        self.upload_folder = upload_folder

    def process(self, file) -> Tuple[dict, int]:
        temp_path, extracted_files = None, []
        try:
            is_valid, message = self.extractor.validate_file(file)
            if not is_valid:
                return {"error": message}, 400
            temp_path = self.extractor.save_uploaded_file(file, self.upload_folder)
            csv_files = self.extractor.extract_zip(temp_path, self.upload_folder)
            if not csv_files:
                return {"error": "Nenhum arquivo CSV/TXT encontrado no ZIP"}, 400
            total_processed, total_errors = 0, 0
            for csv_filename in csv_files:
                csv_path = os.path.join(self.upload_folder, csv_filename)
                try:
                    df = self.transformer.read_csv(csv_path)
                    df = self.transformer.apply_schema(df, DTYPE_MAP)
                    df = self.transformer.normalize_keys(df, KEY_COLUMNS)
                    is_valid, msg, df = self.transformer.validate_data(df)
                    if not is_valid:
                        logger.warning(f"Arquivo {csv_filename}: {msg}")
                        total_errors += 1
                        continue
                    total_processed += self.loader.upsert_dataframe(df, KEY_COLUMNS)
                except Exception as e:
                    logger.error(f"Erro processando {csv_filename}: {str(e)}")
                    total_errors += 1
            return {
                "total_records": total_processed,
                "files_processed": len(csv_files) - total_errors,
                "files_with_errors": total_errors,
            }, 200
        except Exception as e:
            logger.error(f"Erro no pipeline ETL: {str(e)}")
            return {"error": str(e)}, 500
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            for filename in extracted_files:
                filepath = os.path.join(self.upload_folder, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)


# ========== ROTAS DA API ==========


@app.get("/", doc_ui=False)
def index():
    return app.send_static_file("index.html")


@app.post("/api/import", doc_ui=False)
def import_data():
    uploaded_files = request.files.getlist("files")
    if not uploaded_files:
        return jsonify({"error": "Nenhum arquivo fornecido"}), 400
    total_records, success_files, error_files, errors = 0, 0, 0, []
    etl = ETLPipeline(UPLOAD_FOLDER, DB_PATH)
    for file in uploaded_files:
        if file.filename.endswith(".zip"):
            try:
                result, status_code = etl.process(file)
                if status_code == 200:
                    total_records += result.get("total_records", 0)
                    success_files += 1
                else:
                    error_files += 1
                    errors.append(f"'{file.filename}': {result.get('error', 'Erro')}")
            except Exception as e:
                error_files += 1
                errors.append(f"'{file.filename}': {str(e)}")
        else:
            error_files += 1
            errors.append(f"'{file.filename}' não é um ZIP válido.")

    if error_files == 0:
        status_code = 200
    elif success_files > 0:
        status_code = 202
    else:
        status_code = 500
    return (
        jsonify(
            {
                "total_records": total_records,
                "files_processed": success_files,
                "files_with_errors": error_files,
                "errors": errors,
            }
        ),
        status_code,
    )


@app.get("/api/statistics")
def get_statistics():
    query = DataQuery(DB_PATH)
    return jsonify(query.get_statistics()), 200


@app.get("/api/states")
def get_states():
    query = DataQuery(DB_PATH)
    return jsonify(query.get_states_list()), 200


@app.get("/api/search")
@limiter.limit("100 per minute")
def search_addresses():
    cep, logradouro, localidade = (
        request.args.get("cep"),
        request.args.get("logradouro"),
        request.args.get("localidade"),
    )
    cod_uf = request.args.get("cod_uf", type=int)
    limit = min(request.args.get("limit", 100, type=int), 1000)
    offset = request.args.get("offset", 0, type=int)
    group_by_street = request.args.get("group_by_street", "0") == "1"
    query = DataQuery(DB_PATH)
    results, total = query.search_addresses(
        cep, logradouro, localidade, cod_uf, limit, offset, group_by_street
    )
    return (
        jsonify({"results": results, "total": total, "limit": limit, "offset": offset}),
        200,
    )


@app.post("/api/bulk-search")
@limiter.limit("30 per minute")
def bulk_search_addresses():
    data = request.get_json()
    if not data or "filters" not in data:
        return jsonify({"error": "Corpo da requisição inválido."}), 400
    filters = data.get("filters", [])
    limit = min(data.get("limit", 100), 5000)
    offset = data.get("offset", 0)
    group_by_street = data.get("group_by_street", False)
    query = DataQuery(DB_PATH)
    results, total = query.bulk_search_addresses(
        filters, limit, offset, group_by_street
    )
    return (
        jsonify({"results": results, "total": total, "limit": limit, "offset": offset}),
        200,
    )


# ========== FUNCOES AUXILIARES ==========


def search_addresses(
    self,
    cep: Optional[str] = None,
    logradouro: Optional[str] = None,
    localidade: Optional[str] = None,
    cod_uf: Optional[int] = None,
    limit: int = 100,
    offset: int = 0,
    group_by_street: bool = False,
) -> Tuple[List[Dict], int]:
    """Busca endereços com filtros"""
    try:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if (
                group_by_street
                and cep
                and not logradouro
                and not localidade
                and not cod_uf
            ):
                cursor.execute(
                    """
                    SELECT 
                        E.cep,
                        E.nom_tipo_seglogr,
                        E.nom_seglogr,
                        E.dsc_localidade,
                        E.cod_uf,
                        E.cod_municipio,
                        E.num_endereco,
                        E.nom_comp_elem1,
                        E.val_comp_elem1,
                        E.nom_comp_elem2,
                        E.val_comp_elem2,
                        E.nom_comp_elem3,
                        E.val_comp_elem3,
                        UPPER(M.nome) as nome
                    FROM endereco E
                    JOIN municipio M ON E.cod_municipio = M.cod_municipio AND E.cod_uf = M.cod_uf
                    WHERE cep = ?
                    ORDER BY 
                        CASE 
                            WHEN nom_tipo_seglogr LIKE 'RUA%' THEN 1
                            WHEN nom_tipo_seglogr LIKE 'AV%' THEN 2
                            ELSE 3
                        END,
                        nom_seglogr ASC
                    LIMIT 1
                """,
                    (cep,),
                )
                rows = cursor.fetchall()
                total_count = len(rows)
            else:
                where_clauses = []
                params = []

                if cep:
                    where_clauses.append("cep LIKE ?")
                    params.append(f"%{cep}%")

                if logradouro:
                    where_clauses.append(
                        "(nom_tipo_seglogr || ' ' || nom_seglogr) LIKE ?"
                    )
                    params.append(f"%{logradouro}%")

                if localidade:
                    where_clauses.append("M.nome LIKE ? COLLATE NOCASE")
                    params.append(f"%{localidade}%")

                if cod_uf:
                    where_clauses.append("M.cod_uf = ?")
                    params.append(cod_uf)

                where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

                if group_by_street:
                    count_query = f"""
                        SELECT COUNT(*) FROM (
                            SELECT cep, nom_tipo_seglogr, nom_seglogr
                            FROM endereco E
                            JOIN municipio M ON E.cod_municipio = M.cod_municipio AND E.cod_uf = M.cod_uf
                            WHERE {where_sql}
                            GROUP BY cep, nom_tipo_seglogr, nom_seglogr
                        ) AS sub
                    """
                    cursor.execute(count_query, params)
                    total_count = cursor.fetchone()[0]

                    query = f"""
                        SELECT 
                            MIN(E.rowid) as rowid,
                            E.cep,
                            E.nom_tipo_seglogr,
                            E.nom_seglogr,
                            E.nom_comp_elem1,
                            E.val_comp_elem1,
                            E.nom_comp_elem2,
                            E.val_comp_elem2,
                            E.nom_comp_elem3,
                            E.val_comp_elem3,
                            E.dsc_localidade,
                            M.cod_uf,
                            M.cod_municipio,
                            MIN(E.num_endereco) as num_endereco,
                            UPPER(M.nome) as nome
                        FROM endereco E
                        JOIN municipio M ON E.cod_municipio = M.cod_municipio AND E.cod_uf = M.cod_uf
                        WHERE {where_sql}
                        GROUP BY cep, nom_tipo_seglogr, nom_seglogr, dsc_localidade, M.cod_uf, M.cod_municipio
                        ORDER BY cep, nom_seglogr
                        LIMIT ? OFFSET ?
                    """
                    cursor.execute(query, params + [limit, offset])
                    rows = cursor.fetchall()
                else:
                    count_query = f"""
                        SELECT 
                            COUNT(*) 
                        FROM endereco E
                        JOIN municipio M ON E.cod_municipio = M.cod_municipio AND E.cod_uf = M.cod_uf
                        WHERE {where_sql}
                    """
                    cursor.execute(count_query, params)
                    total_count = cursor.fetchone()[0]

                    query = f"""
                        SELECT 
                            E.cep,
                            E.nom_tipo_seglogr,
                            E.nom_seglogr,
                            E.nom_comp_elem1,
                            E.val_comp_elem1,
                            E.nom_comp_elem2,
                            E.val_comp_elem2,
                            E.nom_comp_elem3,
                            E.val_comp_elem3,
                            E.dsc_localidade,
                            M.cod_uf,
                            M.cod_municipio,
                            E.num_endereco,
                            UPPER(M.nome) as nome
                        FROM endereco E
                        JOIN municipio M ON E.cod_municipio = M.cod_municipio AND E.cod_uf = M.cod_uf
                        WHERE {where_sql}
                        ORDER BY cep, nom_seglogr
                        LIMIT ? OFFSET ?
                    """
                    cursor.execute(query, params + [limit, offset])
                    rows = cursor.fetchall()

            results = []
            for row in rows:
                uf_info = UF_INFO.get(
                    row["cod_uf"], {"sigla": "", "estado": "", "regiao": ""}
                )
                tipo = row["nom_tipo_seglogr"] or ""
                nome = row["nom_seglogr"] or ""
                logradouro_full = f"{tipo} {nome}".strip()
                comp_nome = row["nom_comp_elem1"] or ""
                comp_valor = row["val_comp_elem1"] or ""
                comp_nome2 = row["nom_comp_elem2"] or ""
                comp_valor2 = row["val_comp_elem2"] or ""
                complemento = (
                    f"{comp_nome} {comp_valor} {comp_nome2} {comp_valor2}".strip()
                    if comp_nome or comp_valor
                    else ""
                )
                dsc_localidade = row["dsc_localidade"] or ""
                numero = (
                    row["num_endereco"]
                    if "num_endereco" in row.keys() and row["num_endereco"]
                    else ""
                )
                cidade = row["nome"] if "nome" in row.keys() and row["nome"] else ""
                results.append(
                    {
                        "cep": row["cep"],
                        "logradouro": logradouro_full,
                        "numero": numero,
                        "unidade": "",
                        "bairro": dsc_localidade,
                        "cidade": cidade,
                        "complemento": complemento,
                        "uf": uf_info["sigla"],
                        "estado": uf_info["estado"],
                        "regiao": uf_info["regiao"],
                        "ibge": (
                            str(row["cod_municipio"]) if row["cod_municipio"] else ""
                        ),
                        "gia": "",
                        "siafi": "",
                    }
                )
            return results, total_count
    except Exception as e:
        logger.error(f"Erro na busca: {str(e)}")
        return [], 0


def bulk_search_addresses(
    self,
    filters: List[Dict],
    limit: int = 100,
    offset: int = 0,
    group_by_street: bool = False,
) -> Tuple[List[Dict], int]:
    """Busca endereços com base em uma lista de filtros (busca em lote)."""
    try:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            bulk_where_clauses = []
            params = []
            uf_map = {info["sigla"].upper(): code for code, info in UF_INFO.items()}

            for f in filters:
                single_filter_clauses = []

                cep = f.get("cep", "").strip()
                if cep:
                    single_filter_clauses.append("cep LIKE ?")
                    params.append(f"%{cep.replace('-', '')}%")

                logradouro = f.get("logradouro", "").strip()
                if logradouro:
                    single_filter_clauses.append(
                        "(nom_tipo_seglogr || ' ' || nom_seglogr) LIKE ?"
                    )
                    params.append(f"%{logradouro}%")

                localidade = f.get("localidade", "").strip()
                if localidade:
                    single_filter_clauses.append("M.nome LIKE ? COLLATE NOCASE")
                    params.append(f"%{localidade}%")

                uf_sigla = f.get("uf", "").strip().upper()
                if uf_sigla and uf_sigla in uf_map:
                    single_filter_clauses.append("M.cod_uf = ?")
                    params.append(uf_map[uf_sigla])

                if single_filter_clauses:
                    bulk_where_clauses.append(
                        f"({' AND '.join(single_filter_clauses)})"
                    )

            if not bulk_where_clauses:
                return [], 0

            where_sql = " OR ".join(bulk_where_clauses)

            if group_by_street:
                count_query = f"""
                    WITH RankedAddresses AS (
                        SELECT
                            E.cep,
                            ROW_NUMBER() OVER (
                                PARTITION BY E.cep
                                ORDER BY
                                    CASE
                                        WHEN E.nom_tipo_seglogr LIKE 'RUA%' THEN 1
                                        WHEN E.nom_tipo_seglogr LIKE 'AV%' THEN 2
                                        ELSE 3
                                    END,
                                    E.nom_seglogr ASC
                            ) as rn
                        FROM endereco E
                        JOIN municipio M ON E.cod_municipio = M.cod_municipio AND E.cod_uf = M.cod_uf
                        WHERE {where_sql}
                    )
                    SELECT COUNT(*) FROM RankedAddresses WHERE rn = 1
                """
                cursor.execute(count_query, params)
                total_count = cursor.fetchone()[0]

                query = f"""
                    WITH RankedAddresses AS (
                        SELECT
                            E.cep, E.nom_tipo_seglogr, E.nom_seglogr,
                            E.nom_comp_elem1, E.val_comp_elem1, E.nom_comp_elem2, E.val_comp_elem2,
                            E.dsc_localidade, M.cod_uf, M.cod_municipio, E.num_endereco,
                            UPPER(M.nome) as nome,
                            ROW_NUMBER() OVER (
                                PARTITION BY E.cep
                                ORDER BY
                                    CASE
                                        WHEN E.nom_tipo_seglogr LIKE 'RUA%' THEN 1
                                        WHEN E.nom_tipo_seglogr LIKE 'AV%' THEN 2
                                        ELSE 3
                                    END,
                                    E.nom_seglogr ASC
                            ) as rn
                        FROM endereco E
                        JOIN municipio M ON E.cod_municipio = M.cod_municipio AND E.cod_uf = M.cod_uf
                        WHERE {where_sql}
                    )
                    SELECT * FROM RankedAddresses
                    WHERE rn = 1
                    ORDER BY cep, nom_seglogr
                    LIMIT ? OFFSET ?
                """
                cursor.execute(query, params + [limit, offset])
                rows = cursor.fetchall()
            else:
                count_query = f"SELECT COUNT(*) FROM endereco E JOIN municipio M ON E.cod_municipio = M.cod_municipio AND E.cod_uf = M.cod_uf WHERE {where_sql}"
                cursor.execute(count_query, params)
                total_count = cursor.fetchone()[0]

                query = f"""
                    SELECT 
                        E.cep, E.nom_tipo_seglogr, E.nom_seglogr, E.nom_comp_elem1, E.val_comp_elem1,
                        E.nom_comp_elem2, E.val_comp_elem2, E.dsc_localidade, M.cod_uf,
                        M.cod_municipio, E.num_endereco, UPPER(M.nome) as nome
                    FROM endereco E
                    JOIN municipio M ON E.cod_municipio = M.cod_municipio AND E.cod_uf = M.cod_uf
                    WHERE {where_sql}
                    ORDER BY cep, nom_seglogr
                    LIMIT ? OFFSET ?
                """
                cursor.execute(query, params + [limit, offset])
                rows = cursor.fetchall()

            results = []
            for row in rows:
                uf_info = UF_INFO.get(
                    row["cod_uf"], {"sigla": "", "estado": "", "regiao": ""}
                )
                tipo = row["nom_tipo_seglogr"] or ""
                nome = row["nom_seglogr"] or ""
                logradouro_full = f"{tipo} {nome}".strip()
                comp_nome = row["nom_comp_elem1"] or ""
                comp_valor = row["val_comp_elem1"] or ""
                complemento = (
                    f"{comp_nome} {comp_valor}".strip()
                    if comp_nome or comp_valor
                    else ""
                )
                dsc_localidade = row["dsc_localidade"] or ""
                numero = (
                    row["num_endereco"]
                    if "num_endereco" in row.keys() and row["num_endereco"]
                    else ""
                )
                cidade = row["nome"] if "nome" in row.keys() and row["nome"] else ""
                results.append(
                    {
                        "cep": row["cep"],
                        "logradouro": logradouro_full,
                        "numero": numero,
                        "bairro": dsc_localidade,
                        "cidade": cidade,
                        "complemento": complemento,
                        "uf": uf_info["sigla"],
                        "estado": uf_info["estado"],
                        "regiao": uf_info["regiao"],
                        "ibge": (
                            str(row["cod_municipio"]) if row["cod_municipio"] else ""
                        ),
                    }
                )
            return results, total_count
    except Exception as e:
        logger.error(f"Erro na busca em lote: {str(e)}")
        return [], 0


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True, use_reloader=True)
