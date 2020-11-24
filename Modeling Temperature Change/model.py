# -*- coding: utf-8 -*-
# Problem Set 5: Modeling Temperature Change
# Name:
# Collaborators:
# Time:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import re

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAINING_INTERVAL = range(1961, 2000)
TESTING_INTERVAL = range(2000, 2017)

##########################
#    Begin helper code   #
##########################

def standard_error_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d numpy array of values estimated by a linear
            regression model
        model: a numpy array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = np.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]


class Dataset(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Dataset instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature

        f.close()

    def get_daily_temps(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d numpy array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return np.array(temperatures)

    def get_temp_on_date(self, city, month, day, year):
        """
        Get the temperature for the given city at the specified date.

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified date and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year {} is not available".format(year)
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]

##########################
#    End helper code     #
##########################

    def get_yearly_averages(self, cities, years):
        """
        For each year in the given range of years, computes the average of the
        annual temperatures in the given cities.

        Args:
            cities: a list of the names of cities to include in the average
                annual temperature calculation
            years: a list of years to evaluate the average annual temperatures at

        Returns:
            a 1-d numpy array of floats with length = len(years). Each element in
            this array corresponds to the average annual temperature over the given
            cities for a given year.
        """

        # NOTE: TO BE IMPLEMENTED IN PART 4B OF THE PSET
        temps = []
        for year in years:
            average_temp = [np.mean(self.get_daily_temps(city, year)) for city in cities]
            temps.append(sum(average_temp)/len(cities))
        return temps
        
def linear_regression(x, y):
    """
    Calculates a linear regression model for the set of data points.
    
    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points

    Returns:
        (m, b): A tuple containing the slope and y-intercept of the regression line,
                both of which are floats.
    """
    x_avg = np.mean(x)
    y_avg = np.mean(y)
    sum_xy = 0 
    sum_x = 0
    for i in range(len(x)):
        sum_xy += (y[i]-y_avg)*(x[i]-x_avg)
        sum_x += (x[i]-x_avg)**2
    m = sum_xy / sum_x
    b = y_avg - (m * x_avg)
    
    return m, b     

def squared_error(x, y, m, b):
    '''
    Calculates the squared error of the linear regression model given the set
    of data points.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        m: The slope of the regression line
        b: The y-intercept of the regression line


    Returns:
        a float for the total squared error of the regression evaluated on the
        data set
    '''
    y_est = []
    SE = 0
    for i in range(len(y)):
        y_est = m*x[i]+b # plug in each x
        SE += (y_est - y[i])**2
    return SE


def generate_models(x, y, degrees):
    """
    Generates a list of polynomial regression models with degrees specified by
    degrees for the given set of data points

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        degrees: a list of integers that correspond to the degree of each polynomial
            model that will be fit to the data

    Returns:
        a list of numpy arrays, where each array is a 1-d numpy array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    np_coef = []
    for deg in degrees:
        np_coef.append(np.polyfit(x, y, deg))
    return np_coef

def evaluate_models(x, y, models, display_graphs=False):
    """
    For each regression model, compute the R-squared value for this model and
    if display_graphs is True, plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (i.e. the model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        Degree of your regression model,
        R-squared of your model evaluated on the given data points,
        and standard error/slope (if this model is linear).

    R-squared and standard error/slope should be rounded to 4 decimal places.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial
        display_graphs: A boolean whose value specifies if the graphs should be
            displayed

    Returns:
        A list holding the R-squared value for each model
    """
    R_sq = []
    for model in models:
        y_est=[np.polyval(model, x_value) for x_value in x]
        # plug in x to each model and obtain a list of estimated values
        r2 = r2_score(y, y_est)
        R_sq.append(r2)
        if display_graphs:
            plt.figure()
            plt.plot(x, y, 'b.')
            # plot data points
            plt.plot(x, y_est, 'r-')
            # plot modeled lines
            plt.xlabel('Years')
            plt.ylabel('Temperature(Celsius)')
            if len(model) == 2:
                SE = squared_error(x, y, model[0], model[1])
                plt.title('Model with degree 1 and R2 score of ' + str(r2) +\
                          ', the standard error is '+ str(SE))
            else:
                plt.title('Model with degree '+ str(len(model)-1) + ' and R2 score of ' + str(r2))
    return R_sq
            
        

def find_extreme_trend(x, y, length, positive_slope):
    """
    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        length: the length of the interval
        positive_slope: a boolean whose value specifies whether to look for
            an interval with the most extreme positive slope (True) or the most
            extreme negative slope (False)

    Returns:
        a tuple of the form (i, j, m) such that the application of linear (deg=1)
        regression to the data in x[i:j], y[i:j] produces the most extreme
        slope m, with the sign specified by positive_slope and j-i = length.

        In the case of a tie, it returns the first interval. For example,
        if the intervals (2,5) and (8,11) both have slope 3.1, (2,5,3.1) should be returned.

        If no intervals matching the length and sign specified by positive_slope
        exist in the dataset then return None
    """
    slopes = {}
    if positive_slope:
        for i in range(0, len(x)-length+1):
            j = i + length
            slope = generate_models(x[i:j], y[i:j], [1])
            # generate a model with the given i, j
            slopes[i,j]=slope[0][0]
            # the slope of the first model in the list of slope because we only want 
            # linear slopes
        max_slope = max(slopes.values())
        if max_slope < 0:# if the max is negative
            return None
        else:
            for i, j in slopes.keys():
                if abs(slopes[i,j]-max_slope) < 1e-8:
                    return i, j, max_slope
    else:
        for i in range(0, len(x)-length+1):
            j = i + length
            slope = generate_models(x[i:j], y[i:j], [1])
            slopes[i,j]=slope[0][0]
        min_slope = min(slopes.values())
        if min_slope > 0:
            return None
        else:
            for i, j in slopes.keys():
                if abs(slopes[i,j]-min_slope) < 1e-8:
                    return i, j, min_slope


def find_all_extreme_trends(x, y):
    """
    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        
    Returns:
        a list of tuples of the form (i,j,m) such that the application of linear
        regression to the data in x[i:j], y[i:j] produces the most extreme
        positive OR negative slope m, and j-i=length. 

        The returned list should have len(x) - 1 tuples, with each tuple representing the
        most extreme slope and associated interval for all interval lengths 2 through len(x).
        If there is no positive or negative slope in a given interval length L (m=0 for all
        intervals of length L), the tuple should be of the form (0,L,None).

        The returned list should be ordered by increasing interval length. For example, the first 
        tuple should be for interval length 2, the second should be for interval length 3, and so on.

        If len(x) < 2, return an empty list
    """
    all_extreme_trends = []
    for n in range(2,len(x)+1):
        if find_extreme_trend(x, y, n, positive_slope=True) != None: #if there is a positive slope
            i, j, slope_pos = find_extreme_trend(x, y, n, positive_slope=True)
            if find_extreme_trend(x, y, n, positive_slope=False) != None: #if there is a negative slope
                m, n, slope_neg = find_extreme_trend(x, y, n, positive_slope=False)
                if slope_pos > abs(slope_neg) + 1e-8:# compare both, if positive is much bigger
                # use 1e-8 to avoid rounding errors
                    all_extreme_trends.append((i, j, slope_pos))
                else:#if negative slope is more extreme
                    all_extreme_trends.append((m, n, slope_neg))
            else:
                all_extreme_trends.append((i, j, slope_pos))
        else:#if no positive slope
            if find_extreme_trend(x, y, n, positive_slope=False) != None: #if there is a negative slope
                m, n, slope_neg = find_extreme_trend(x, y, n, positive_slope=False)
                all_extreme_trends.append((m, n, slope_neg))
            else:# if slope == 0 for all slopes
                all_extreme_trends.append((0, n, None))
    return all_extreme_trends


def rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d numpy array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    sq_error = 0
    for n in range(len(y)):
        sq_error += (y[n]-estimated[n])**2
    return np.sqrt(sq_error/len(y))


def evaluate_models_testing(x, y, models, display_graphs=False):
    """
    For each regression model, compute the RMSE for this model and if
    display_graphs is True, plot the test data along with the model's estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points.

    RMSE should be rounded to 4 decimal places.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N test data sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N test data sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial.
        display_graphs: A boolean whose value specifies if the graphs should be
            displayed

    Returns:
        A list holding the RMSE value for each model
    """
    RMSEs = []
    for model in models:
        y_est=[np.polyval(model, x_value) for x_value in x]
        RMSE = np.round(rmse(y, y_est), 4)
        RMSEs.append(RMSE)
        if display_graphs:
            plt.figure()
            plt.plot(x, y, 'b.')
            plt.plot(x, y_est, 'r-')
            plt.xlabel('Years')
            plt.ylabel('Temperature(Celsius)')
            if len(model) == 2:
                SE = squared_error(x, y, model[0], model[1])
                plt.title('Model with degree 1 and RMSE score of ' + str(RMSE) +\
                          ', the standard error is '+ str(SE))
            else:
                plt.title('Model with degree '+ str(len(model)-1) + ' and RMSE score of ' + str(RMSE))
    return RMSEs



if __name__ == '__main__':
    pass
    ##################################################################################
    # Problem 4A: DAILY TEMPERATURE
    # record = Dataset('data.csv')
    # x=[n for n in range(1961, 2017)]
    # y=[record.get_temp_on_date('SAN FRANCISCO', 12, 25, n) for n in range(1961, 2017)]
    # models = generate_models(x, y, [1])
    # r2 = evaluate_models(x, y, models, display_graphs = True)

    ##################################################################################
    # Problem 4B: ANNUAL TEMPERATURE
    # record = Dataset('data.csv')
    # x=[n for n in range(1961, 2017)]
    # y=record.get_yearly_averages(['SAN FRANCISCO'], range(1961, 2017))
    # models = generate_models(x, y, [1])
    # r2 = evaluate_models(x, y, models, display_graphs = True)
    
    # Concept Questions:
    # 4.1 What difference does choosing a specific day to plot the data versus 
    # calculating the yearly average have on the goodness of fit of the model? 
    # Interpret the results.
    #   There might be seasonal fluctuations in the temperature on sepecific days.
    #   The yearly average eliminates this problem and could fit the line better
    
    
    # 4.2 Why do you think these graphs are so noisy?
    #   The sample data is too small and might have large deviation

    ##################################################################################
    # Problem 5B: INCREASING TRENDS
    # record = Dataset('data.csv')
    # averages = record.get_yearly_averages(['TAMPA'], range(1961, 2017))
    # i, j , slope = find_extreme_trend([n for n in range(1961, 2017)], averages, 30, True)
    # x = range(1961+i, 1961+j)
    # y = averages[i:j]
    # models = generate_models(x, y, [1])
    # r2 = evaluate_models(x, y, models, display_graphs = True)
    # print(r2)
    # (1, 31, 0.04648393050274639)
    # r2=0.31645807958420613

    ##################################################################################
    # Problem 5C: DECREASING TRENDS
    # record = Dataset('data.csv')
    # averages = record.get_yearly_averages(['TAMPA'], range(1961, 2017))
    # i, j, slope = find_extreme_trend([n for n in range(1961, 2017)], averages, 15, False)
    # x = range(1961+i, 1961+j)
    # y = averages[i:j]
    # models = generate_models(x, y, [1])
    # r2 = evaluate_models(x, y, models, display_graphs = True)
    # (9, 24, -0.032569507715513396)
    # r2=0.05901893034513761
    #   The temperature is more significantly increasing than decreasing because the
    #   window examined and r2 score is larger for temperature rising.

    ##################################################################################
    # Problem 5D: ALL EXTREME TRENDS
    # record = Dataset('data.csv')
    # x = [n for n in range(1961, 2017)]
    # averages = record.get_yearly_averages(['TAMPA'], range(1961, 2017))
    # print(find_all_extreme_trends(x, averages))
    # 5.4 How many intervals showed a regression with a more extreme positive slope? 
    # Negative slope?
    #   
    #   54 positive slopes and 1 negative slope
    #
    # 5.5 Suppose you'd like point the citizens of Tampa to the vast number of intervals 
    # with a linear regression that has a more extreme positive slope as evidence that 
    # temperatures are rising. Is this, by itself, a convincing argument? If you were 
    # "Turn Down the AC", what is a statistical argument you could use to counter this?
    #   
    #   There are much more positive slopes than negative slopes, which implies that
    #   the temperature has been steadily increasing over the past 55 years.
    #   "Turn Down the AC" might argue that there might be some extreme statistical outlier
    #   that interferes with the result since the positive trend is less and less significant
    #   when the interval is longer.

    ##################################################################################
    # Problem 6B: PREDICTING
    # record = Dataset('data.csv')
    # TRAINING_AVERAGES = record.get_yearly_averages(CITIES, TRAINING_INTERVAL)
    # models = generate_models(TRAINING_INTERVAL, TRAINING_AVERAGES, range(2, 10))
    # print(evaluate_models(TRAINING_INTERVAL, TRAINING_AVERAGES, models, True))
    # r2_values = [0.7645960656796663, 0.7676197090712091, 0.7676654505990008, 0.7753191523757654, 
    # 0.7752477959064231, 0.7751733811652233, 0.7750997318465064, 0.7750265931372543]
    #
    # 6.1 How do these models compare to each other in terms of R^2 and fitting the data?
    #   The model fits better and better when the degree is higher, which is reflected in
    #   the higher r2 score.
    # TESTING_AVERAGES = record.get_yearly_averages(CITIES, TESTING_INTERVAL)
    # evaluate_models_testing(TESTING_INTERVAL, TESTING_AVERAGES, models, True)
    # 6.4 Which model performed the best and how do you know? If this is different from the 
    # training performance in the previous section, explain why.
    # 
    #   The linear model performed the best because the RMSE value is the lowest.
    #   This is different from the training performance because in the higher degree 
    #   polynomial, the y values would change very drastically even with a small change
    #   in x, which does not correspond to the actual behavior of national climate.
    #
    # 6.5 If we had generated the models using the data from Problem 4B (i.e. the average 
    # annual temperature of San Francisco) instead of the national annual average over the 
    # 22 cities, how would the prediction results on the national data have changed?
    #   
    #   The RMSE value would be much larger, with the estimated value much smaller than the 
    #   actual temperature because the temperature in San Francisco is generally lower than
    #   the national average.

    ##################################################################################
