# Importación de librerías
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from datetime import date, timedelta

# Configuración de la página de Streamlit
st.set_page_config(page_title="ETFs y Acciones - Análisis Completo", layout="wide")

# Título de la aplicación
st.title("📊 Análisis Financiero Completo de ETFs y Acciones")

# Parámetros de consulta en la barra lateral
st.sidebar.header("Configuración del análisis")

# Entrada del símbolo del ETF o acción
ticker = st.sidebar.text_input("Ingrese el símbolo del ETF o Acción (Ej: QQQ, VOO, TSLA)", value="QQQ")

# Fechas de inicio y fin para la consulta de datos
start_date = st.sidebar.date_input("Fecha de inicio", value=pd.to_datetime("2014-01-01"))
end_date = st.sidebar.date_input("Fecha de fin", value=pd.to_datetime(date.today()))

# Función para calcular rendimiento y volatilidad
def annualized_performance(returns, periods_per_year=252):
    if len(returns) == 0:
        return np.nan, np.nan  # Retorna NaN si no hay datos suficientes

    mean_return = np.mean(returns) * periods_per_year
    volatility = np.std(returns) * np.sqrt(periods_per_year)
    return mean_return, volatility

# Función para calcular el cambio de precio y rendimiento en distintos periodos
def display_price_changes(ticker_obj):
    try:
        # Descargar el historial completo de precios
        data = ticker_obj.history(period="1y")
        if data.empty:
            st.warning("No se encontraron datos de precios para calcular los cambios.")
            return

        # Obtener el precio actual
        current_price = data["Close"].iloc[-1]

        # Función para calcular rendimiento en porcentaje
        def calculate_change(days):
            if len(data) < days:
                return "N/A", "N/A"
            past_price = data["Close"].iloc[-days]
            price_change = current_price - past_price
            percent_change = (price_change / past_price) * 100
            return price_change, percent_change

        # Calcular cambios para diferentes periodos
        changes = {
            "Periodo": ["Diario", "Semanal", "Mensual", "Trimestral", "Semestral", "Anual"],
            "Cambio ($)": [],
            "Cambio (%)": [],
        }
        periods = [1, 5, 21, 63, 126, 252]  # Aproximado: días bursátiles

        for days in periods:
            price_change, percent_change = calculate_change(days)
            changes["Cambio ($)"].append(f"{price_change:.2f}" if price_change != "N/A" else "N/A")
            changes["Cambio (%)"].append(f"{percent_change:.2f}%" if percent_change != "N/A" else "N/A")

        # Crear DataFrame para mostrar resultados
        changes_df = pd.DataFrame(changes)

        # Mostrar tabla estilizada
        st.subheader("📊 Cambios de Precio y Rendimiento")
        st.write(
            changes_df.style
            .set_table_styles(
                [
                    {"selector": "thead th", "props": [("background-color", "#4CAF50"), ("color", "white"), ("font-size", "16px")]},
                    {"selector": "tbody td", "props": [("font-size", "14px"), ("text-align", "center")]},
                    {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#f2f2f2")]},
                ]
            )
            .set_properties(**{"text-align": "center"})
            .to_html(index=False),  # Ocultar índice
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Error al calcular cambios de precio: {e}")

# Función para mostrar el precio actual
def display_current_price(ticker_obj):
    try:
        # Obtener el precio actual desde el atributo info
        info = ticker_obj.info
        current_price = info.get("currentPrice", None)
        currency = info.get("currency", "USD")

        if current_price is not None:
            st.subheader(f"💵 Precio Actual de {ticker.upper()}")
            st.metric(label="Precio Actual", value=f"{current_price:.2f} {currency}")
        else:
            st.warning("No se pudo obtener el precio actual para este activo.")

    except Exception as e:
        st.error(f"Error al obtener el precio actual: {e}")

# Función para mostrar fundamentales
def display_fundamentals(ticker_obj):
    try:
        info = ticker_obj.info

        # Extracto de fundamentales clave
        market_cap = info.get("marketCap", None)
        pe_ratio = info.get("trailingPE", None)
        dividend_yield = info.get("dividendYield", None)
        beta = info.get("beta", None)
        revenue = info.get("totalRevenue", None)
        profit_margin = info.get("profitMargins", None)

        # Crear un DataFrame con los datos
        fundamentals_data = {
            "Indicador": [
                "Capitalización de Mercado (B)",
                "P/E Ratio",
                "Rendimiento por Dividendo (%)",
                "Beta",
                "Ingresos Totales (B)",
                "Margen de Ganancia (%)",
            ],
            "Valor": [
                f"{market_cap / 1e9:.2f} B" if market_cap else "N/A",
                f"{pe_ratio:.2f}" if pe_ratio else "N/A",
                f"{dividend_yield * 100:.2f} %" if dividend_yield else "N/A",
                f"{beta:.2f}" if beta else "N/A",
                f"{revenue / 1e9:.2f} B" if revenue else "N/A",
                f"{profit_margin * 100:.2f} %" if profit_margin else "N/A",
            ],
        }
        fundamentals_df = pd.DataFrame(fundamentals_data)

        # Mostrar tabla estilizada
        st.subheader("📊 Resumen de Fundamentales")
        st.write(
            fundamentals_df.style
            .set_table_styles(
                [
                    {"selector": "thead th", "props": [("background-color", "#4CAF50"), ("color", "white"), ("font-size", "16px")]},
                    {"selector": "tbody td", "props": [("font-size", "14px"), ("text-align", "center")]},
                    {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#f2f2f2")]},
                ]
            )
            .set_properties(**{"text-align": "center"})
            .to_html(index=False),  # Aquí ocultamos el índice al renderizar HTML
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Error al obtener fundamentales: {e}")

# Parámetros para la simulación Monte Carlo
num_simulations = st.sidebar.slider("Número de Simulaciones", min_value=100, max_value=1000, value=500)
num_days = st.sidebar.slider("Días a Simular", min_value=30, max_value=365, value=180)

# Función para calcular los rendimientos diarios
def calculate_returns(data, column):
    return data[column].pct_change().dropna()

# Función para realizar la simulación Monte Carlo
def monte_carlo_simulation(start_price, mu, sigma, num_days, num_simulations):
    np.random.seed(42)
    simulations = []
    for _ in range(num_simulations):
        daily_returns = np.random.normal(mu, sigma, num_days)
        price_series = [start_price]
        for ret in daily_returns:
            price_series.append(price_series[-1] * (1 + ret))
        simulations.append(price_series)
    return np.array(simulations).T

# Función para mostrar fundamentales
def display_fundamentals(ticker_obj):
    try:
        info = ticker_obj.info

        # Validar si info está disponible
        if not info:
            st.error("No se encontraron fundamentales para el activo seleccionado.")
            return

        # Extracto de fundamentales clave
        market_cap = info.get("marketCap", None)
        pe_ratio = info.get("trailingPE", None)
        dividend_yield = info.get("dividendYield", None)
        beta = info.get("beta", None)
        revenue = info.get("totalRevenue", None)
        profit_margin = info.get("profitMargins", None)

        # Crear un DataFrame con los datos
        fundamentals_data = {
            "Indicador": [
                "Capitalización de Mercado (B)",
                "P/E Ratio",
                "Rendimiento por Dividendo (%)",
                "Beta",
                "Ingresos Totales (B)",
                "Margen de Ganancia (%)",
            ],
            "Valor": [
                f"{market_cap / 1e9:.2f} B" if market_cap else "N/A",
                f"{pe_ratio:.2f}" if pe_ratio else "N/A",
                f"{dividend_yield * 100:.2f} %" if dividend_yield else "N/A",
                f"{beta:.2f}" if beta else "N/A",
                f"{revenue / 1e9:.2f} B" if revenue else "N/A",
                f"{profit_margin * 100:.2f} %" if profit_margin else "N/A",
            ],
        }
        fundamentals_df = pd.DataFrame(fundamentals_data)

        # Mostrar tabla
        st.subheader("📊 Resumen de Fundamentales")
        st.dataframe(fundamentals_df)

    except Exception as e:
        st.error(f"Error al obtener fundamentales: {e}")

# Botón para realizar la consulta
if st.sidebar.button("Consultar y Analizar"):
    try:
        # Obtener el objeto Ticker de yfinance
        ticker_obj = yf.Ticker(ticker)

        # Mostrar el precio actual
        display_current_price(ticker_obj)

        # Resumen del ETF o Acción
        st.subheader(f"🔍 Resumen de {ticker.upper()}")
        info = ticker_obj.info  # Información general del activo

        # Mostrar cambios de precio y rendimientos
        display_price_changes(ticker_obj)

        if info:
            name = info.get("shortName", "N/A")
            sector = info.get("sector", "N/A")
            industry = info.get("industry", "N/A")
            description = info.get("longBusinessSummary", "No disponible")
            currency = info.get("currency", "N/A")
            exchange = info.get("exchange", "N/A")
            
            st.write(f"**Nombre completo:** {name}")
            st.write(f"**Sector:** {sector}")
            st.write(f"**Industria:** {industry}")
            st.write(f"**Moneda:** {currency}")
            st.write(f"**Intercambio:** {exchange}")
            st.write(f"**Descripción:** {description}")

        # Mostrar fundamentales
        display_fundamentals(ticker_obj)

        # Descargar los datos del ETF o Acción
        data_etf = ticker_obj.history(start=start_date, end=end_date)

        # Descargar datos de comparación (SPY)
        data_spy = yf.download("SPY", start=start_date, end=end_date)

        # Determinar la columna de precios a usar
        price_column = "Adj Close" if "Adj Close" in data_etf.columns else "Close"

        if not data_etf.empty and not data_spy.empty:

            # Convertir índices a naive datetime (sin zona horaria)
            data_etf.index = data_etf.index.tz_localize(None)
            data_spy.index = data_spy.index.tz_localize(None)

            # Mostrar los datos en tablas
            st.subheader(f"Datos Históricos del ETF/Acción: {ticker.upper()}")
            st.dataframe(data_etf.tail())

            # Cálculo de rendimientos diarios
            returns_etf = calculate_returns(data_etf, price_column)
            returns_spy = calculate_returns(data_spy, "Adj Close")

            # Gráfico de comparación de precios ajustados
            st.subheader(f"📉 Comparación del Precio Ajustado: {ticker.upper()} vs SPY")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data_etf[price_column], label=f"{ticker.upper()} - Precio")
            ax.plot(data_spy['Adj Close'], label="SPY - Precio Ajustado", linestyle='--')
            ax.set_title(f"Comparación de Precios: {ticker.upper()} vs SPY")
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Precio ($)")
            ax.legend()
            st.pyplot(fig)

            # Cálculo del rendimiento acumulado
            cumulative_returns_etf = (1 + returns_etf).cumprod()
            cumulative_returns_spy = (1 + returns_spy).cumprod()

            # Gráfico de comparación del rendimiento acumulado
            st.subheader(f"📊 Rendimiento Acumulado: {ticker.upper()} vs SPY")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(cumulative_returns_etf, label=f"{ticker.upper()} - Rendimiento Acumulado")
            ax.plot(cumulative_returns_spy, label="SPY - Rendimiento Acumulado", linestyle='--')
            ax.set_title(f"Rendimiento Acumulado: {ticker.upper()} vs SPY")
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Rendimiento Acumulado")
            ax.legend()
            st.pyplot(fig)

            # Campana de Gauss para los últimos 5 años
            st.subheader(f"📈 Campana de Gauss: Cambios Porcentuales Diarios (Últimos 5 Años)")
            last_5_years = date.today() - timedelta(days=5*365)
            data_last_5_years = data_etf[data_etf.index >= pd.to_datetime(last_5_years)]
            returns_last_5_years = calculate_returns(data_last_5_years, price_column)

            # Ajuste de distribución normal
            mu, std = norm.fit(returns_last_5_years)

            # Gráfico de distribución con ajuste normal
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.histplot(returns_last_5_years, bins=50, kde=False, stat="density", ax=ax, label="Datos")
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            ax.plot(x, p, 'r--', label=f"Normal fit: μ={mu:.4f}, σ={std:.4f}")
            ax.set_title(f"Distribución de Cambios Porcentuales Diarios: {ticker.upper()} (Últimos 5 Años)")
            ax.set_xlabel("Cambios Porcentuales Diarios")
            ax.set_ylabel("Densidad")
            ax.legend()
            st.pyplot(fig)

            # Simulación Monte Carlo
            start_price = data_etf[price_column].iloc[-1]
            simulated_prices = monte_carlo_simulation(start_price, mu, std, num_days, num_simulations)

            # Graficar simulación Monte Carlo
            st.subheader("📈 Simulación Monte Carlo")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(simulated_prices, alpha=0.1, color="blue")
            ax.set_title(f"Simulación Monte Carlo: {ticker.upper()} (Último Precio: ${start_price:.2f})")
            ax.set_xlabel("Días Futuros")
            ax.set_ylabel("Precio Simulado ($)")
            st.pyplot(fig)

            
        else:
            st.error("No se encontraron datos para el ETF o la Acción en las fechas seleccionadas.")

    except Exception as e:
        st.error(f"Error al descargar los datos: {e}")

else:
    st.write("Introduzca un símbolo de ETF o Acción y presione 'Consultar y Analizar' para comenzar.")

# Pie de página
st.sidebar.markdown("Desarrollado con ❤️ por Python GPT")