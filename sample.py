from flask_restful import Resource

# pip install pandas-datareader
# load data from yahoo finance
import numpy as np
import pandas as pd

# pip install cvxopt
# pip install portfolioopt
import portfolioopt as popt
from rd_api_datareader.APIDataReader.DataReader import DataReader
from rd_database.models.assets import Asset, PortfolioAsset
from rd_config.app import db


class PortfolioOptimization(Resource):

    def optimize(self, data, portfolio_version_id):
        # data = request.get_json()

        strDateFrom =  data['dateFrom']
        strDateTo = data['dateTo']
        # access to database in MySql

        assets = db.session.query(PortfolioAsset).filter_by(version_id=portfolio_version_id).join(Asset).all()

        # check if user exists
        if len(assets) == 0:
            return {}

        stocks = []
        for asset in assets:
            stocks.append(asset.asset.ticker)

        # request data from datareader and store them in prices (pandas data form)
        req = {'tickers': stocks, 'start': strDateFrom, 'end':  strDateTo }
        prices = DataReader().read_data_pd_format(req)

        # convert daily stock prices into daily returns
        returns = prices.pct_change(-1)
        # drop infs and nans
        returns = returns.replace([np.inf, -np.inf], np.nan)
        returns = returns.dropna()
        # estimate mean return and covariance matrix
        ExpRet = returns.mean()
        CovRet = returns.cov()

        # compute portfolios with min var and max return
        minvar_port = popt.min_var_portfolio(CovRet)
        minret = max(np.dot(minvar_port, ExpRet), 0)
        maxret = max(ExpRet)

        # calculate range of attainable returns for optimal portfolios
        mus = np.linspace(start=minret, stop=maxret, num=100)

        # calculate efficient frontier
        portfolios = [popt.markowitz_portfolio(CovRet, ExpRet, mu) for mu in mus]

        dts, portfolio_indexes = self.get_value_indexes(prices, portfolios)
        # expected returns
        exRet = [np.dot(portfolio, ExpRet) * 252 for portfolio in portfolios]

        # expected volatility
        exVol = [np.sqrt(np.dot(np.dot(portfolio, CovRet), portfolio) * 252) for portfolio in portfolios]
        return {
                   'assets': stocks,
                   'portfolios': pd.Series(portfolios).to_json(orient='values'),
                   'annualizedExpectedReturns': exRet,
                   'annualizedExpectedVolatility': exVol,
                   'dates': dts.tolist(),
                   'portfolio_indexes': np.array(portfolio_indexes).tolist()
               }

    @staticmethod
    def get_value_indexes(prices, portfolios):
        interval = 5  # once a week
        precision = 1
        # retrievs dates
        dts = np.flip(np.array(prices.index.values.tolist()), 0)[::interval]/ 1000000

        prices = np.flip(np.asarray(prices).astype(np.float), 0)[::interval]

        portfolio_indexes = []
        for portfolio in portfolios:
            portfolio_prices = np.dot(prices, portfolio)
            portfolio_indexes.append(np.around(100 * portfolio_prices / portfolio_prices[0], precision))

        return dts, portfolio_indexes

