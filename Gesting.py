# app.py
# Streamlit app para subir PDFs de facturas, extraer datos (número, fecha, proveedor,
# base imponible, IVA, total) y generar resumenes/exportar CSV.
#
# Requisitos: streamlit, pdfplumber, pandas
# Instalar: pip install streamlit pdfplumber pandas

import re
import io
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import pdfplumber
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Resumen Facturas PDF", layout="wide")


############################
# Utilidades de parsing
############################

AMOUNT_RE = re.compile(r"(?<!\w)(?:\d{1,3}(?:[.,]\d{3})*[.,]\d{2}|\d+[.,]\d{2})(?!\w)")
DATE_RE = re.compile(
    r"(?:(?:\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})|(?:\d{4}[\/\-.]\d{1,2}[\/\-.]\d{1,2}))"
)
INVOICE_NO_RE = re.compile(
    r"(?:N(?:º|o|úmero|úm)\.?\s*[:\-\s]?\s*([A-Za-z0-9\-\/]+))|(?:Factura(?:\s*N(?:º|o|úmero)?)?\s*[:\-\s]?\s*([A-Za-z0-9\-\/]+))",
    flags=re.IGNORECASE,
)
BASE_RE = re.compile(
    r"(?:Base imponible|Base imponible\s*[:\-]?\s*|Base\s*[:\-]?\s*)(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})",
    flags=re.IGNORECASE,
)
IVA_LINE_RE = re.compile(
    r"(?:IVA|I\.V\.A|Impuesto sobre el valor añadido)\s*(?:[:\-]?\s*)?(?:(\d{1,2}(?:[.,]\d)?)\%?)?.*?(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})",
    flags=re.IGNORECASE,
)
TOTAL_RE = re.compile(
    r"(?:Total a pagar|TOTAL A PAGAR|Total factura|Importe total|TOTAL|Total\s*[:\-]?\s*)(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})",
    flags=re.IGNORECASE,
)
SUPPLIER_RE = re.compile(
    r"(?:Proveedor|Empresa|Razón social|Emitida por|From|De)\s*[:\-]?\s*([A-ZÁÉÍÓÚÑ0-9\w \-\.,&]+)",
    flags=re.IGNORECASE,
)

CURRENCY_SYMBOLS = ["€", "EUR", "EUR.", "€.", "$", "USD"]


def normalize_amount_text(s: str) -> str:
    """Normalize common separators to a dot decimal, remove spaces and currency symbols."""
    if s is None:
        return ""
    s = s.strip()
    # Remove currency symbols
    for symbol in CURRENCY_SYMBOLS:
        s = s.replace(symbol, "")
    # Remove spaces
    s = s.replace(" ", "")
    # If format like 1.234,56 -> convert to 1234.56
    # If format like 1,234.56 (english) -> remove commas
    # Heuristic: if comma exists and there are exactly two digits after last comma -> treat comma as decimal sep
    if "," in s and "." in s:
        # Determine which looks like decimal sep (last separator)
        last_comma = s.rfind(",")
        last_dot = s.rfind(".")
        if last_comma > last_dot:
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        # If there are exactly two digits after last comma, treat as decimal sep
        if re.search(r",\d{2}$", s):
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        s = s.replace(",", "")
    # Remove any non-numeric except dot and minus
    s = re.sub(r"[^\d\.\-]", "", s)
    return s


def parse_amount(s: Optional[str]) -> Optional[float]:
    if s is None or s == "":
        return None
    try:
        norm = normalize_amount_text(s)
        if norm == "":
            return None
        return float(norm)
    except Exception:
        return None


def find_first_amount_in_text(text: str) -> Optional[float]:
    m = AMOUNT_RE.search(text)
    if not m:
        return None
    return parse_amount(m.group(0))


def extract_text_from_pdf_bytes(b: bytes) -> str:
    """Extrae texto concatenando las páginas. Maneja errores de pdfplumber."""
    text_parts = []
    try:
        with pdfplumber.open(io.BytesIO(b)) as pdf:
            for p in pdf.pages:
                try:
                    page_text = p.extract_text() or ""
                except Exception:
                    page_text = ""
                text_parts.append(page_text)
    except Exception as e:
        # Intentar una extracción mínima con PyPDF2 podría añadirse, pero aquí devolvemos cadena vacía
        st.warning(f"Error abriendo PDF: {e}")
        return ""
    return "\n".join(text_parts)


def extract_invoice_fields_from_text(text: str) -> Dict[str, Optional[str]]:
    """Intenta detectar campos claves en el texto de una factura usando regex heurísticos."""
    res = {
        "invoice_no": None,
        "date": None,
        "supplier": None,
        "base": None,
        "iva_rate": None,
        "iva_amount": None,
        "total": None,
        "raw": text[:1000],  # snippet for debugging / display
    }

    # Buscar número de factura
    m = INVOICE_NO_RE.search(text)
    if m:
        invoice_no = m.group(1) or m.group(2)
        if invoice_no:
            res["invoice_no"] = invoice_no.strip()

    # Buscar fecha (primera coincidencia)
    m = DATE_RE.search(text)
    if m:
        raw_date = m.group(0)
        parsed = try_parse_date(raw_date)
        if parsed:
            res["date"] = parsed.strftime("%Y-%m-%d")
        else:
            res["date"] = raw_date

    # Buscar proveedor (primer match del patrón supplier, o heurística: primeras líneas con mayúsculas)
    m = SUPPLIER_RE.search(text)
    if m:
        supplier = m.group(1).strip()
        # limpiar terminadores raros
        supplier = re.sub(r"\s{2,}", " ", supplier)
        if len(supplier) > 0:
            res["supplier"] = supplier[:80]

    if not res["supplier"]:
        # heurística: primera línea larga que no contenga 'FACTURA' ni 'NIF' y tenga letras
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            for ln in lines[:8]:
                if len(ln) > 3 and "FACTUR" not in ln.upper() and not any(char.isdigit() for char in ln[:5]):
                    res["supplier"] = ln[:80]
                    break

    # Buscar base imponible
    m = BASE_RE.search(text)
    if m:
        res["base"] = m.group(1)

    # Buscar IVA: buscar todas las líneas que contengan 'IVA' y tomar la más probable
    iva_candidates = []
    for match in IVA_LINE_RE.finditer(text):
        rate = match.group(1)
        amount = match.group(2)
        iva_candidates.append((rate, amount))
    if iva_candidates:
        # Usar el último IVA detectado (suele ser el total IVA)
        rate, amount = iva_candidates[-1]
        if rate:
            res["iva_rate"] = rate.strip().replace(",", ".")
        res["iva_amount"] = amount

    # Buscar total - tomar la última coincidencia plausible
    totals = []
    for match in TOTAL_RE.finditer(text):
        totals.append(match.group(1))
    if totals:
        res["total"] = totals[-1]
    else:
        # fallback: buscar el último amount grande en el doc
        all_amounts = AMOUNT_RE.findall(text)
        if all_amounts:
            res["total"] = all_amounts[-1]

    # Si hay base pero no IVA amount, intentar calcular con total-base
    if res.get("base") and res.get("total") and not res.get("iva_amount"):
        base_v = parse_amount(res["base"])
        tot_v = parse_amount(res["total"])
        if base_v is not None and tot_v is not None:
            iva_calc = tot_v - base_v
            if iva_calc is not None and iva_calc != 0:
                res["iva_amount"] = f"{iva_calc:.2f}"

    # Normalize amounts to numeric strings (dot decimal)
    for k in ["base", "iva_amount", "total"]:
        if res.get(k):
            parsed = parse_amount(res[k])
            if parsed is not None:
                res[k] = f"{parsed:.2f}"
            else:
                res[k] = None

    # iva_rate: normalize
    if res.get("iva_rate"):
        res["iva_rate"] = res["iva_rate"].replace(",", ".")
        try:
            # keep as percent string like '21' or '4.0'
            _ = float(res["iva_rate"])
            res["iva_rate"] = res["iva_rate"]
        except Exception:
            res["iva_rate"] = None

    return res


def try_parse_date(s: str) -> Optional[datetime]:
    s = s.strip()
    # Replace common separators
    s = s.replace(".", "/").replace("-", "/")
    parts = s.split("/")
    try_formats = []
    if len(parts) == 3:
        # dd/mm/yyyy or yyyy/mm/dd or dd/mm/yy
        try_formats = ["%d/%m/%Y", "%Y/%m/%d", "%d/%m/%y", "%m/%d/%Y", "%m/%d/%y"]
    else:
        try_formats = ["%Y-%m-%d", "%d-%m-%Y"]
    for fmt in try_formats:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    # fallback: try to extract numbers
    numbers = re.findall(r"\d{1,4}", s)
    if len(numbers) >= 3:
        # attempt to build a date
        y, m, d = None, None, None
        # heuristics: if one number has 4 digits -> year
        for num in numbers:
            if len(num) == 4:
                y = int(num)
                break
        if not y:
            # assume last is year
            y = int(numbers[-1])
        # pick first two for day/month
        try:
            possible = [int(n) for n in numbers if len(n) <= 2][:2]
            if len(possible) == 2:
                d, m = possible[0], possible[1]
                return datetime(y, m, d)
        except Exception:
            return None
    return None


############################
# Procesamiento de PDFs
############################

def process_uploaded_pdf(file_uploader_buffer: io.BytesIO) -> Dict[str, Optional[str]]:
    """Dado un archivo (BytesIO) retorna diccionario con campos extraidos."""
    try:
        raw_bytes = file_uploader_buffer.read()
    except Exception as e:
        st.error(f"Error leyendo archivo: {e}")
        return {}
    text = extract_text_from_pdf_bytes(raw_bytes)
    if not text:
        # intentar lectura como texto plano
        try:
            text = raw_bytes.decode("utf-8", errors="ignore")
        except Exception:
            text = ""
    fields = extract_invoice_fields_from_text(text)
    # guardar el nombre del archivo en raw para referencia
    fields["filename"] = getattr(file_uploader_buffer, "name", "uploaded.pdf")
    return fields


############################
# Streamlit UI
############################

st.title("📄 Resumenador de Facturas (PDF) — Streamlit")
st.write(
    "Sube uno o varios PDFs de facturas. La app intentará extraer Número, Fecha, Proveedor, Base, IVA y Total. Puedes corregir manualmente antes de añadir al resumen."
)

uploaded_files = st.file_uploader(
    "Sube archivos PDF de facturas (puedes seleccionar varios)",
    type=["pdf"],
    accept_multiple_files=True,
)

# Contenedor para resultados añadidos
if "facturas" not in st.session_state:
    st.session_state.facturas = []

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Facturas detectadas / por procesar")
    if uploaded_files:
        for uploaded in uploaded_files:
            st.markdown("---")
            st.subheader(f"Archivo: {uploaded.name}")
            with st.expander("Vista previa del texto extraído y resultados automáticos", expanded=False):
                try:
                    bytes_io = io.BytesIO(uploaded.read())
                    # reset pointer
                    bytes_io.seek(0)
                    fields = process_uploaded_pdf(bytes_io)
                except Exception as e:
                    st.error(f"Error procesando {uploaded.name}: {e}")
                    fields = {}
                # Mostrar snippet
                snippet = fields.get("raw", "") if fields else ""
                if snippet:
                    st.text_area("Snippet del texto extraído (primera parte)", snippet, height=180)
                else:
                    st.info("No se ha extraído texto legible automáticamente.")

                # Mostrar campos detectados y permitir edición
                with st.form(key=f"form_{uploaded.name}", clear_on_submit=False):
                    invoice_no = st.text_input(
                        "Número de factura", value=fields.get("invoice_no") if fields else "", key=f"no_{uploaded.name}"
                    )
                    date = st.text_input("Fecha (YYYY-MM-DD)", value=fields.get("date") if fields else "", key=f"date_{uploaded.name}")
                    supplier = st.text_input(
                        "Proveedor / Empresa", value=fields.get("supplier") if fields else "", key=f"supplier_{uploaded.name}"
                    )
                    base = st.text_input(
                        "Base imponible (ej. 1234.56)", value=fields.get("base") if fields else "", key=f"base_{uploaded.name}"
                    )
                    iva_rate = st.text_input(
                        "IVA (%)", value=fields.get("iva_rate") if fields else "", key=f"iva_rate_{uploaded.name}"
                    )
                    iva_amount = st.text_input(
                        "IVA (importe)", value=fields.get("iva_amount") if fields else "", key=f"iva_amount_{uploaded.name}"
                    )
                    total = st.text_input(
                        "Total (importe)", value=fields.get("total") if fields else "", key=f"total_{uploaded.name}"
                    )
                    # Tipo default heurístico: gasto si proveedor detectado (empresa emisora) - el usuario puede cambiar
                    default_tipo = "gasto"
                    tipo = st.selectbox(
                        "Tipo de operación",
                        options=["gasto", "ingreso"],
                        index=0 if default_tipo == "gasto" else 1,
                        key=f"tipo_{uploaded.name}",
                    )

                    add_button = st.form_submit_button("Añadir factura al resumen")

                    if add_button:
                        # Parsear y validar cantidades
                        parsed_base = parse_amount(base)
                        parsed_iva_amount = parse_amount(iva_amount)
                        parsed_total = parse_amount(total)

                        # Si falta total y hay base+iva -> calcular
                        if parsed_total is None and parsed_base is not None and parsed_iva_amount is not None:
                            parsed_total = round(parsed_base + parsed_iva_amount, 2)

                        # Si falta IVA amount y hay base+iva_rate -> calcular
                        if parsed_iva_amount is None and parsed_base is not None and iva_rate:
                            try:
                                rate = float(iva_rate.replace(",", "."))
                                parsed_iva_amount = round(parsed_base * rate / 100.0, 2)
                            except Exception:
                                parsed_iva_amount = None

                        # Si falta base but total and iva_amount exist -> calculate base
                        if parsed_base is None and parsed_total is not None and parsed_iva_amount is not None:
                            parsed_base = round(parsed_total - parsed_iva_amount, 2)

                        factura = {
                            "filename": uploaded.name,
                            "invoice_no": invoice_no or None,
                            "date": date or None,
                            "supplier": supplier or None,
                            "base": f"{parsed_base:.2f}" if parsed_base is not None else None,
                            "iva_rate": iva_rate or None,
                            "iva_amount": f"{parsed_iva_amount:.2f}" if parsed_iva_amount is not None else None,
                            "total": f"{parsed_total:.2f}" if parsed_total is not None else None,
                            "tipo": tipo,
                        }

                        st.session_state.facturas.append(factura)
                        st.success(f"Factura añadida: {uploaded.name}")

    else:
        st.info("No has subido archivos aún.")

with col2:
    st.header("Resumen y tabla general")
    if not st.session_state.facturas:
        st.info("Aún no hay facturas añadidas. Añádelas desde la columna izquierda.")
    else:
        df = pd.DataFrame(st.session_state.facturas)

        # Normalización/Converión de tipos numéricos
        def to_float_col(s):
            try:
                return float(normalize_amount_text(str(s))) if s is not None else None
            except Exception:
                return None

        df["base_num"] = df["base"].apply(to_float_col)
        df["iva_num"] = df["iva_amount"].apply(to_float_col)
        df["total_num"] = df["total"].apply(to_float_col)

        # Totales
        total_gastado = df.loc[df["tipo"] == "gasto", "total_num"].sum(min_count=1)
        total_ingresado = df.loc[df["tipo"] == "ingreso", "total_num"].sum(min_count=1)
        total_iva_soportado = df.loc[df["tipo"] == "gasto", "iva_num"].sum(min_count=1)
        total_iva_repercutido = df.loc[df["tipo"] == "ingreso", "iva_num"].sum(min_count=1)
        beneficio = (total_ingresado or 0.0) - (total_gastado or 0.0)

        # Mostrar DataFrame paginado/filtrable
        with st.expander("Tabla de facturas (editar / eliminar filas)", expanded=True):
            st.dataframe(
                df[
                    [
                        "filename",
                        "invoice_no",
                        "date",
                        "supplier",
                        "base",
                        "iva_rate",
                        "iva_amount",
                        "total",
                        "tipo",
                    ]
                ].fillna(""),
                use_container_width=True,
            )

            # Botón para eliminar todas
            if st.button("Eliminar todas las facturas"):
                st.session_state.facturas = []
                st.experimental_rerun()

        st.markdown("### Totales")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Total gastado", f"{(total_gastado or 0.0):.2f} €")
            st.metric("Total IVA soportado", f"{(total_iva_soportado or 0.0):.2f} €")
        with col_b:
            st.metric("Total ingresado", f"{(total_ingresado or 0.0):.2f} €")
            st.metric("Total IVA repercutido", f"{(total_iva_repercutido or 0.0):.2f} €")

        st.markdown(f"**Beneficio (ingresos - gastos):** {beneficio:.2f} €")

        # Mostrar desglose por proveedor (opcional)
        if st.checkbox("Mostrar desglose por proveedor"):
            by_supplier = (
                df.groupby("supplier", dropna=False)
                .agg(
                    total_gastado=("total_num", lambda x: x[df.loc[x.index, "tipo"] == "gasto"].sum()),
                    total_ingresado=("total_num", lambda x: x[df.loc[x.index, "tipo"] == "ingreso"].sum()),
                    iva_total=("iva_num", "sum"),
                    count=("filename", "count"),
                )
                .fillna(0)
                .reset_index()
            )
            st.dataframe(by_supplier, use_container_width=True)

        # Export CSV del resumen completo
        csv_buffer = io.StringIO()
        export_df = df[
            [
                "filename",
                "invoice_no",
                "date",
                "supplier",
                "base",
                "iva_rate",
                "iva_amount",
                "total",
                "tipo",
            ]
        ].fillna("")
        export_df.to_csv(csv_buffer, index=False, sep=",")
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        st.download_button(
            label="📥 Descargar resumen (CSV)",
            data=csv_bytes,
            file_name=f"resumen_facturas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

st.markdown("---")
st.caption("Nota: La extracción se basa en heurísticas. Revisa siempre los datos extraídos y corrige si fuera necesario antes de añadir al resumen.")
