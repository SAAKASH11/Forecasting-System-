import pandas as pd
import numpy as np
import logging
import click
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import product
import warnings

# Prophet import with fallback
try:
    from prophet import Prophet
except ImportError:
    Prophet = None

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# -------------------------
# Data Loading and Preprocessing
# -------------------------
def load_and_preprocess(filepath):
    logging.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath, parse_dates=['Date'])
    required_cols = {'Product_Code', 'Warehouse', 'Product_Category', 'Date', 'Order_Demand'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input data must contain columns: {required_cols}")

    # Clean Order_Demand:
    # 1. Remove commas
    # 2. Convert '(x)' to negative numbers
    def parse_order_demand(x):
        x = str(x).replace(',', '').strip()
        if x.startswith('(') and x.endswith(')'):
            x = '-' + x[1:-1]
        return float(x)

    df['Order_Demand'] = df['Order_Demand'].apply(parse_order_demand)

    # Sort data by Product_Code and Date
    df = df.sort_values(['Product_Code', 'Date']).reset_index(drop=True)
    return df

# -------------------------
# ARIMA Model with Grid Search
# -------------------------
def arima_grid_search(train_series, p_values, d_values, q_values):
    best_aic = np.inf
    best_order = None
    best_model = None

    for order in product(p_values, d_values, q_values):
        try:
            model = SARIMAX(train_series, order=order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            model_fit = model.fit(disp=False)
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_order = order
                best_model = model_fit
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("ARIMA grid search failed to find a suitable model.")

    logging.info(f"Best ARIMA order for product: {best_order} with AIC: {best_aic:.2f}")
    return best_model

def fit_arima_model(train_series, p_values=[0,1,2], d_values=[0,1], q_values=[0,1,2]):
    best_model = arima_grid_search(train_series, p_values, d_values, q_values)
    class ModelWrapper:
        def __init__(self, arima_res):
            self.arima_res_ = arima_res
    return ModelWrapper(best_model)

def forecast_arima(model, train_series, periods=14):
    last_date = train_series.index[-1]
    forecast_res = model.arima_res_.get_forecast(steps=periods)
    forecast = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
    forecast.index = future_dates
    conf_int.index = future_dates
    return forecast, conf_int

# -------------------------
# Prophet Model
# -------------------------
def fit_prophet_model(train_df):
    if Prophet is None:
        raise ImportError("Prophet is not installed. Please install it with `pip install prophet`.")
    m = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
    m.fit(train_df)
    return m

def forecast_prophet(model, periods=14):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    forecast_future = forecast.tail(periods)
    forecast_series = forecast_future.set_index('ds')['yhat']
    conf_int = forecast_future.set_index('ds')[['yhat_lower', 'yhat_upper']]
    return forecast_series, conf_int

# -------------------------
# Evaluation Metrics
# -------------------------
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_forecast(true_series, forecast_series):
    common_index = true_series.index.intersection(forecast_series.index)
    true_vals = true_series.loc[common_index]
    pred_vals = forecast_series.loc[common_index]
    mae = mean_absolute_error(true_vals, pred_vals)
    rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
    mape = mean_absolute_percentage_error(true_vals, pred_vals)
    return mae, rmse, mape

# -------------------------
# Plotting and Reporting
# -------------------------
def plot_forecast(train_series, forecast, conf_int, product_code, model_name):
    plt.figure(figsize=(12,6))
    plt.plot(train_series.index, train_series.values, label='Historical', color='blue')
    plt.plot(forecast.index, forecast.values, label='Forecast', color='orange')
    plt.fill_between(conf_int.index, conf_int.iloc[:,0], conf_int.iloc[:,1], color='orange', alpha=0.3, label='Confidence Interval')
    plt.title(f'Product {product_code} Sales Forecast ({model_name})')
    plt.xlabel('Date')
    plt.ylabel('Order Demand')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(3)
    plt.close()

def generate_pdf_report(forecast_df, eval_df, product_code, model_name):
    filename = f'report_product_{product_code}_{model_name}.pdf'
    with PdfPages(filename) as pdf:
        # Plot forecast
        plt.figure(figsize=(10,6))
        product_forecast = forecast_df[forecast_df['Product_Code'] == product_code]
        plt.plot(product_forecast['date'], product_forecast['forecast_sales'], label='Forecast')
        plt.fill_between(product_forecast['date'], product_forecast['conf_lower'], product_forecast['conf_upper'], alpha=0.3)
        plt.title(f'Forecast for Product {product_code} ({model_name})')
        plt.xlabel('Date')
        plt.ylabel('Order Demand')
        plt.legend()
        plt.grid(True)
        pdf.savefig()
        plt.close()

        # Add evaluation table as text page
        eval_row = eval_df[eval_df['Product_Code'] == product_code]
        plt.figure(figsize=(8, 2))
        plt.axis('off')
        text = eval_row.to_string(index=False)
        plt.text(0.1, 0.5, text, fontsize=12)
        pdf.savefig()
        plt.close()
    logging.info(f"Saved PDF report: {filename}")

# -------------------------
# Main CLI Application
# -------------------------
@click.command()
@click.option('--datafile', required=True, help='Path to the CSV dataset file')
@click.option('--model', default='arima', type=click.Choice(['arima', 'prophet']), help='Forecasting model to use')
@click.option('--forecast_days', default=14, help='Number of days to forecast')
@click.option('--save_forecast', is_flag=True, help='Save forecast results to CSV')
@click.option('--save_evaluation', is_flag=True, help='Save evaluation metrics to CSV')
@click.option('--plot', is_flag=True, help='Plot forecasts for each product')
@click.option('--generate_reports', is_flag=True, help='Generate PDF reports for each product')
def main(datafile, model, forecast_days, save_forecast, save_evaluation, plot, generate_reports):
    try:
        df = load_and_preprocess(datafile)
    except Exception as e:
        logging.error(f"Failed to load or preprocess data: {e}")
        return

    product_codes = df['Product_Code'].unique()
    logging.info(f"Forecasting {len(product_codes)} products using {model.upper()} model.")

    all_forecasts = []
    all_evaluations = []

    for product_code in product_codes:
        logging.info(f"Processing product {product_code}")
        product_data = df[df['Product_Code'] == product_code].copy()
        product_data = product_data.sort_values('Date')
        train_series = product_data.set_index('Date')['Order_Demand']

        if len(train_series) < 10:
            logging.warning(f"Not enough data for product {product_code}, skipping.")
            continue

        try:
            if model == 'arima':
                arima_model = fit_arima_model(train_series)
                forecast, conf_int = forecast_arima(arima_model, train_series, periods=forecast_days)
            else:
                if Prophet is None:
                    logging.error("Prophet is not installed. Install it with `pip install prophet`.")
                    return
                train_df = product_data.rename(columns={'Date':'ds', 'Order_Demand':'y'})[['ds','y']]
                prophet_model = fit_prophet_model(train_df)
                forecast, conf_int = forecast_prophet(prophet_model, periods=forecast_days)
        except Exception as e:
            logging.error(f"Model fitting or forecasting failed for product {product_code}: {e}")
            continue

        # Evaluation on last forecast_days of historical data (ARIMA only)
        eval_mae = eval_rmse = eval_mape = np.nan
        if model == 'arima' and len(train_series) > forecast_days:
            eval_start = train_series.index[-forecast_days]
            eval_true = train_series.loc[eval_start:]
            try:
                pred_res = arima_model.arima_res_.get_prediction(start=eval_start)
                eval_pred = pred_res.predicted_mean
                eval_mae, eval_rmse, eval_mape = evaluate_forecast(eval_true, eval_pred)
            except Exception as e:
                logging.warning(f"Evaluation failed for product {product_code}: {e}")

        all_evaluations.append({
            'Product_Code': product_code,
            'Model': model,
            'MAE': eval_mae,
            'RMSE': eval_rmse,
            'MAPE': eval_mape
        })

        forecast_df = pd.DataFrame({
            'Product_Code': product_code,
            'date': forecast.index,
            'forecast_sales': forecast.values,
            'conf_lower': conf_int.iloc[:, 0].values,
            'conf_upper': conf_int.iloc[:, 1].values
        })

        all_forecasts.append(forecast_df)

        if plot:
            plot_forecast(train_series, forecast, conf_int, product_code, model.upper())

        if generate_reports:
            combined_forecast_df = pd.concat(all_forecasts, ignore_index=True)
            eval_df = pd.DataFrame(all_evaluations)
            generate_pdf_report(combined_forecast_df, eval_df, product_code, model.upper())

    if all_forecasts:
        result_df = pd.concat(all_forecasts, ignore_index=True)
        if save_forecast:
            output_file = 'product_forecasts.csv'
            result_df.to_csv(output_file, index=False)
            logging.info(f"Forecasts saved to {output_file}")
    else:
        logging.warning("No forecasts generated.")

    if all_evaluations and save_evaluation:
        eval_df = pd.DataFrame(all_evaluations)
        eval_file = 'forecast_evaluation.csv'
        eval_df.to_csv(eval_file, index=False)
        logging.info(f"Evaluation metrics saved to {eval_file}")

    logging.info("Forecasting process completed.")

if __name__ == '__main__':
    main()
