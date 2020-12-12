import quandl
import pandas as pd

class GetData():

    def __init__(self, start_date = "2010-01-01", end_date = "2019-01-01", api = '7y4Y2Lk2-####_Ms97GU', company_code = 'AMZN', TDC = 'WIKI'):
        self.start_date = start_date
        self.end_date = end_date
        self.api = api
        self.company_code = company_code
        self.TDC = TDC
        print(self.TDC + '/' + self.company_code)
        #self.column = column


    def get_data(self, reset_index = False):
        quandl.ApiConfig.api_key = self.api
        #print('WIKI/' + self.company_code)
        self.data = quandl.get(str(self.TDC + '/' + self.company_code), start_date = self.start_date, end_date = self.end_date)
        if (reset_index == True):
            self.data = self.data.reset_index()
        else:
            pass
        pass

    def return_data(self, column = None):
        self.get_data()
        
        self.column = column
        if (self.column == None):
            return self.data

        else:
            return self.data[self.column]
        
