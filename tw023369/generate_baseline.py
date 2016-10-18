''' Takes  the sales history for one item at all locations and creates a baseline.

This code is time and loc agnostic.  After the history periods have been selected the process runs against that history.

The location can be a store or an aggregated level( assuming a history has previously been created)'''

import logging
mylogger = logging.getLogger(__name__)

import pprint

import pandas as pd
from pandas.tseries.offsets import Week, Day
import numpy as np

from functools import partial, reduce
from itertools import imap, repeat, starmap, ifilter
from datetime import date
import sys
from collections import namedtuple, OrderedDict
#-------------- Load Modules----------------------------
# xxx.importIO shares a common IO configuration across all modules
# that require IO
import HAVI.JDA_demand.core as core
from HAVI.JDA_demand.common import *
from HAVI.JDA_demand.core import monkeypatch_method
import HAVI.JDA_demand.seasonality.application as seasonality
import HAVI.JDA_demand.holidays.application as holidays
import HAVI.JDA_demand.events as events
import HAVI.JDA_demand.proxy as proxy
import HAVI.JDA_demand.baseline.smoothing as smoothing
import HAVI.JDA_demand.baseline.base_price as base_price
import HAVI.JDA_demand.baseline.cleanse as cleanse

import HAVI.JDA_demand.io.hist_io as hist_io
import HAVI.JDA_demand.baseline.adu as adu

from HAVI.JDA_demand.constants import *
from HAVI.JDA_demand.io.conditions import  *

from HAVI.debug import dummy_dump



# from HAVI.JDA_demand.algos.smooth_item_array import smooth_a_history
# from HAVI.JDA_demand.algos.rolling_moment import isGoodDay

from singledispatch import  singledispatch
from HAVI.JDA_demand.configurations import (Full, Incremental, Seasonality, Holiday,
                                                      Weeks, Days, DOW)

#-------------------Load the standard commnaline argument hanl

@monkeypatch_method(pd.DataFrame)
def get_top_row(self, index=True):
    if self.empty:
        return None
    else:
        return self.itertuples(index=index).next() 

def exportIO(data_model):
    ''' Provide all submodules that perform IO access to the passed in data model
    
    If we go to a parallel model we may need to explicitly create the data model 
    and then export it 
    '''
    seasonality.importIO(data_model)
    holidays.importIO(data_model) 
    events.importIO(data_model)
    proxy.importIO(data_model)
    common_importIO(data_model)
    hist_io.importIO(data_model)
    adu.importIO(data_model)
    cleanse.importIO(data_model)
    #Now make IO availbalbe in this modulea s well
    global IO
    IO =data_model
    return

pp = pprint.PrettyPrinter(indent  = 2, width = 50,)


# def get_hist_for_geogs(config, item, geog_cut):   
#       geogs  = PerfectBetween(geog_cut)
#       hist_key  = hist_io.build_shard(item=item,
#                                     geogs=geogs,
#                                     config=config)
#       return hist_key
# 
# 
#         
def filter_geogs(config, geog_df, geogs, dfus):
    ''' A function to reduce the set of geogs that need to be processed for history 
    
    The values returned are:
    -- a reduced dataframe with the same columns as geogs_df but only rows for geogs that are true candidates for
       baseline generation
    -- geog_eceptions a dataframe that has one row for each geog removed from the filtered array together with a
       reason code.
    '''
    
    fitered_geogs = geog_df
    geog_exceptions = pd.DataFrame( columns = ['geog', 'condition'])
    return fitered_geogs, geog_exceptions


    
    
def get_hist_window(desired_start, desired_end, first_hist_date, week_periods = False):
    ''' Return the effective window .for a geog or process based on available history
    
    The output is expressed as offsets from the start of history this enables the use of array indexing in the smoothing process
    
    '''
    start_diff  = (desired_start- 
                               first_hist_date)\
                            .days
    if start_diff <= 0:
        start_week = 0
        start_day = 0
    else:
        start_week, start_day  = divmod(start_diff,
                                      7)
    # The end date is open     
    if desired_end == pd.NaT:
        end_week = 1000
        end_day = 1000
    else:
        end_week, end_day =  divmod ((desired_end- 
                                                     first_hist_date)
                                          .days,
                                          7)
        # end week is going to be used in a slice so add1 to include this week
        end_week += 1
    # If the data is held in week buckets then the start and end days mean nothing
    if week_periods:
        start_day = end_day = 0    
    return start_week, start_day,  end_week, end_day




#
    
def get_effective_hist_window_for_geogs(config, hist_df, geogs_1, process_config=None):
    ''' When we smooth history we should only use period this item was on sale at the store.
    
    For most items this will be the whole history period. However the store could have been permanently closed or the 
    item moved out of participation before the end of the history window. Similarly with store opening or items going on sale.
    
    We start with the  first and last date of history for a geog as found in geog_hist_dates
    
    We are looking for the first and last dates in the history period for each geography that is between store open close and 
    NOSALE flag is not set.
    
    
    '''
    # Must preseve the geog order in case we use this as the driver for by geog 
    # looping over bare numpy arrays
    res  = OrderedDict()
    for geog in geogs_1.itertuples():
        
        geog_start  = geog.hist_start_date
        geog_end = min( geog.hist_end_date,
                        geog.close_date)
        if geog_end < config.hist_end_date:
            hist_slice =(geog.Index, 
                         slice(geog_end,
                               config.hist_end_date ))
            geog_flags  = hist_df.loc[hist_slice,'flags'].pipe(none_set,[P_CLOSED, P_NOSALE])
            geog_end  = max(geog_end, 
                             geogs_flags[geog_flags].index.get_level_values('bday').max())
            
        res[geog.Index] = GeogWindow(*get_hist_window(geog_start,
                                               geog_end,
                                               config.hist_start_date,
                                               process_config.hist_period =='W') )
            
    return res
    
    

        
   

def conform_to_dates( config, hist_df,  process_config,): 
    ''' Return history with all geogs aligned on the full history period.
    
    If the column flags was not in the source dataframe it will be added. 
    After processing the flags column will have no null values and all added rows flagged as FILL
    However we also use evidence of actual sales to override the fill value'''
    idx  =pd.MultiIndex.from_product([ hist_df.index.levels[0],
                                      pd.date_range(start = config.hist_start_date, 
                                                      end =config.hist_end_date, 
                                                     freq = process_config.hist_period)],
                                     names  = ['geog', 'bday'])
    
        
        
        
    hist2 = hist_df.reindex(idx,)
    hist2.loc[:, 'flags'] = hist2.loc[:, 'flags'].fillna(0).astype('uint16')
    
        # Check for actual sales
    hist2.loc[~hist2.index.isin(hist_df.index)&
                    hist2.sales.isnull(), 'flags'] = (hist2.loc[(~hist2.index.isin(hist_df.index)&
                                                                    hist2.sales.isnull()), 'flags']
                                                            .pipe(set_flag, P_FILL_HISTORY))
         
    return hist2
    
#
def mark_known_events(config, hist_df, event_df):
    ''' Set flagon eqch day in history there is a recoreded event
     
    Foe masking baseline we recognise two differnt exclusions
    -- A promotion, which we expect to affect sales  that is atypical for  non promoted period
    -- A general exclusion probably created by a ademand planner. An example would be a weather event or construction
     
    All promotions are indicated by isPromo flag being set.
    '''
  
    event_df['isPromo'] = event_df.isPromo.fillna(0)
    hist_df = hist_df.join(event_df, how = 'left').sort_index()
    cond = hist_df.isPromo == 1
    hist_df.loc[cond,'flags'] = (hist_df.loc[cond,
                                    'flags']
                                .pipe(set_flag, P_PROMO))
      
    cond1 = hist_df.isPromo == 0
    hist_df.loc[cond1,'flags'] = (hist_df.loc[cond1,
                                    'flags']
                                .pipe(set_flag, P_EXCLUDE))
     
    hist_df = hist_df.drop('isPromo', axis=1)
        
    return hist_df

def apply_seas_holiday_index(config,
                             process_config,
                             grp,
                             func,
                             params,
                             geogs_with_hist,
                             agg_hierarchy,
                             bread, 
                             start= None, 
                             end=None ):
    ''' Collect the seasonality/holiday index and apply to each day.
    
    Function has to accomplish three tasks:
    
    1) Determine which seasonality/holiday group iapplies at each geog
    2) retrieve the best index to apply to each geography.
    
    Determining which seasonality/holiday group starts with the item seasonality groups and then applies any geog level 
    overrides.  Logic assumes that overides are a) few, and b) mostly the same alternate seasonality group.
    
    Therefore approach is:
    1) Collect the best index for the item_seasonality/holiday group for all geogs.
    2) find all alternate sesonality/holiday groups
    3) For each alternate sesonslity/holiday group 
        3a) find the best index for [all|specified] geogs
        3b) replace the item ndex with the alt index for the speciified geogs
    4) explode the final set of indexes down to the individual day
    
    All the index collection / manipulation routines are in the seasonality or holiday package
    
    As the routine is aused for both seasonality and holiday we pass a parameter name [seasonality_group or holiday_group 
    and a function call on the correct package

    '''
    
    #get the defulat seasonality group for this item
    home_group = params.get_default_params()[0][grp]
    
    other_groups =[]
    
    for geog in geogs_with_hist.reset_index().itertuples(index=False):
        
        other_group = params.get_params(geog)[grp]
        if other_group != home_group:
            other_groups.append((geog.geog, other_group))
    
    home_geogs = geogs_with_hist.copy()
    if other_groups:
        home_geogs  = home_geogs.loc[~home_geogs.index.isin([geog for geog,_ in other_groups ])]
    #get the vlaues for 
    res_idx = func(config, process_config, home_group, home_geogs, agg_hierarchy, bread, params, start=start, end=end)
    if other_groups:
        other_groups  =pd.DataFrame.from_tuples(other_groups, columns  = ['geog', 'other_group'])  
        
        other_groups_set  = set(other_groups.other_group)
        res = [pd.Series()]
        for other_group in other_groups_set:
            other_group_geogs =  geogs_with_hist.loc[geogs_with_hist.index
                                                                .isin(other_groups.loc[other_groups[grp]==other_group,'geog']),
                                                            :]
            res.append(func(config, process_config, iorther_group, other_group_geogs, agg_hierarchy, bread,params, start=start, end=end))
        override_idx = pd.concat(res)
        
        # now replace all the original entries with the override entries
        # this is a blind update . 
        res_idx.update(override_idx)
      
    return res_idx

@singledispatch
def manage_seasonality(config, process_config, hist_df, params, geog_sales, geogs_with_hist, agg_hierarchy, bread):
    raise NotImplementedError(bread() + ( "Unknown configuration foo baseline, %r" % config.__class__ ))

@manage_seasonality.register(Full)
@manage_seasonality.register(Incremental)
@manage_seasonality.register(Holiday)
def manage_seasonality_normal(config, process_config, hist_df, params, geog_sales, 
                              geogs_with_hist, agg_hierarchy, bread,start='hist_start_date', end='hist_end_date'):

    ''' Collect the sesonality index and apply to each day.
    
  
    '''
    
    seas_idx = apply_seas_holiday_index(config, 
                                      process_config,
                                      'seasonality_group', 
                                      seasonality.get_seasonality_index, 
                                      params,            
                                      geogs_with_hist, 
                                      agg_hierarchy, 
                                      bread, 
                                      addUnits = True,
                                      start= start, end = end )
 
    res=  hist_df.assign(seasonality_idx =seas_idx)
    if addUnits:
        res  = res.assign(deseasonalized_sales  = lambda x: x.sales/x.seasonality_idx)
    return res

@manage_seasonality.register(Seasonality)
def manage_seasonality_seasonality(config, process_config, hist_df, params, geog_sales, geogs_with_hist, agg_hierarchy, bread):
    ''' There is no seasonality index application  when running baseline to calculate seasonality
    
    This is a no op.
    '''
        
    return hist_df

def manage_holidays(config,
                    process_config,
                    hist_df,
                    params,
                    geog_sales,
                    geogs_with_hist,
                    agg_hierarchy,
                    bread):
    ''' We will mark all holidays  so they can be excluded from smoothing
    
    We will need to get this data againlater when we publish holiday index.
    For now we just mark thew day we will calculate index values for both hisatory and future at the end.
    '''
    
    defined_holidays= holidays.get_holiday_master( )
    reduced_agg_hierarchy = agg_hierarchy.loc[agg_hierarchy.geog.isin(geogs_with_hist.index)]
    holiday_df  = holidays.get_holiday_occurences(config,reduced_agg_hierarchy, start  = 'hist_start_date', end = 'hist_end_date')
    holiday_df = (holiday_df.merge(defined_holidays.loc[defined_holidays.use_in_smoothing == 0,:], 
                                  left_on ='holiday', 
                                  right_index =True, 
                                  how  = 'inner')
                            .set_index(['geog', 'bday'])
                            .sort_index() )
    
    cond  = holiday_df.index.intersection(hist_df.index)
    hist_df.loc[cond,'flags'] = hist_df.loc[cond,'flags'].pipe(set_flag, P_HOLIDAY)
    return hist_df
             

@singledispatch
def _filter_overlay_periods(config, period):
    ''''select all overly periods the lie whole or in part within the history period'''
    return period.end_date >= config.hist_start_date  or period.start_date <= config.hist_end_date

@_filter_overlay_periods.register(Incremental)
def _filter_overlay_periods_incremental(config,
                            period):
    '''For incremental we will not run the overlay process if the window is not in the publish window'''
    return period.end_date >= config.publish_start_date  or period.start_date <= config.hist_end_date
    
            
        

def get_overlay_periods(process_config, config,  bread):
    ''' Get all the overlay calendar periods for which the processs wll run. 
    
    The return value is two date ranges 
    -- The overlay publish window, derived directly from the overlay calendar
    -- the wider window that repreents the history period that will be considered in 
       finding good observations.
       
    In both cases the windows are limited by the total history window.  
    
    If there is no intersection between the overlay publish window and the process publish window 
    there is no overlay smoothing for this window. 
    
    
    '''
   
    overlay_calendar = config.overlay_calendar
    if not overlay_calendar:
        res= []
    else:
        periods = IO.exec_query('overlay', 
                                        calendar = Eq(overlay_calendar) )
        if periods.empty:
            #logger.warning(bread() + ' " No overlay calendar entries found matching calendar name %r' % overlay_calendar)
            res = []
        else:
            periods['end_date'] = pd.Timestamp('2010-01-01')
            if (periods.units =='W').any():
                periods.loc[periods.units =='W','end_date'] = (periods.loc[periods.units =='W','start_date'] + 
                                                               periods.loc[periods.units =='W','periods'].apply(Week) - 
                                                               Day() )
            if (periods.units =='D').any():
                periods.loc[periods.units =='D','end_date'] = (periods.loc[periods.units =='D','start_date'] + 
                                                               periods.loc[periods.units =='D','periods'].apply(Day) - Day() )
            periods['overlay_hist_start'] = periods['start_date'] -Week(config.overlay_look_back) + Day()
            periods['overlay_hist_start'] = periods['overlay_hist_start'].apply(lambda x: max(x,config.hist_start_date))
            periods['overlay_hist_end']   =   periods['end_date'] -Week(config.overlay_look_fwd) - Day()
            periods['overlay_hist_end']   =   periods['overlay_hist_end'].apply(lambda x: max(x, config.hist_end_date) )
            filter_func  = partial(_filter_overlay_periods, config)
            res  = []
            for period in ifilter(filter_func, periods.itertuples(index=False)):
                # Get the offsets for the history that will be consumed
                hist_window  = get_hist_window(period.overlay_hist_start,
                                               period.overlay_hist_end,
                                               config.hist_start_date,
                                               week_periods = process_config.week_periods)
                # get the offsets for when to apply the overly back over the original smoothing
                overlay_window  = get_hist_window(period.start_date,
                                                  period.end_date,
                                                  config.hist_start_date,
                                                  week_periods = process_config.week_periods)
                # OverlayWindow holds the offsets for both the use history and apply overlay periods
                res.append( OverlayWindow( *(list(hist_window) + list(overlay_window)) ) )
        return res                
                                       
def exclude_zeros_rule(config, params, geog_sales_class):
    ''' for each geog, deterrmines whether, for this item, we are to consider zeros sales, in the absence of any other
    flags to be a good day in smoothing and down strema proceses.
    
    STUB Process: Always say a zero is a bad day
    '''
    return  pd.Series(True ,index = geog_sales_class.geog)       
        
        
                                        
    

def process_baseline_for_geogs( item, 
                               process_config, 
                               config  = None, 
                               cut  = None,
                               event_handle= None,
                               geogs= None,
                               agg_array = None,
                               params =None,
                               bread=None,
                               dumper=None,
                               **kargs):
    ''' Mainline function for creating the baseline  for a group of geographies.
    
    The results of the multiple function will be aggregated into a single dataframe 
    for the final stages of generating the adu values.
    
    This function is written a =s sequence of steps The config parameter, which encodes the type of baseline 
    will determine what processing happens at each step based on the class of config.
    
    Inside the scope of this process we are manipulating arrays of hist data to create a baseline 
    The hist_df dataframe is indexed and sorted on on geog, bday. That must be an invarient under all transformations.
    
    The dataframe geogs_with_hist is indexed and sorted on geog.  The geogs in geoegs_with_hist is the exact same set as the
    geogs in the index of hist_df.  These conditions are also true for  geog_windows and geogs_sales_class.
    
    Any breakdown in this disciplne  will result in parmatwers and mayb baselines being assiociated with the wrong geog.
    '''
    bread.push('GeogCut', str(cut))
    here = bread()
    bread.pop()
    # We will need these flag sets to determine which flags to use to maks out history observations
    
    flag_sets  = get_flag_sets()
    # retrieve history
    
    hist_df, geog_hist_dates  = hist_io.get_hist_shard(cut[1])
    dumper(hist_df, 'hist_df', step = 1)
    #If this itam has not been baselined before flags will not be be in the hist dataframe
    if 'flags' not in hist_df.columns:
        hist_df['flags']  = pd.Series(0, index = hist_df.index, dtype = 'uint16')
  
    if mylogger.level == logging.DEBUG:
        print hist_df.info()    # collect information about geogs
        
       
    hist_df, geogs_with_hist, exception_geogs, dfus = cleanse.cleanse_hist(config,
                                                                            process_config,
                                                                            item,
                                                                            hist_df,
                                                                            geogs,
                                                                            geog_hist_dates,
                                                                            params,
                                                                            bread,
                                                                            dumper,
                                                                            cut  = cut)   
    dumper(hist_df, 'hist_df', step=2)
    dumper(geogs_with_hist, 'geogs_with_hist', step=2)
    geog_events = events.get_events_for_geogs(config,
                                              event_handle,
                                              cut)
    if not geog_events.empty:
        hist_df = mark_known_events(config,
                                    hist_df,
                                    geog_events)
        
        dumper(hist_df, 'hist_df', step=3)
        
    #logger.info(here+' Finished prep')
    # True up the array to a uniform size and pass pack the reshape parameters
    hist_df = conform_to_dates( config, hist_df, process_config,)
    #logger.info(here+' Conformed history dataframe')
    dumper(hist_df, 'hist_df', step=4)
    geog_windows = get_effective_hist_window_for_geogs(config, 
                                                       hist_df,
                                                       geogs_with_hist,
                                                       process_config=process_config)
    #logger.info(here + ' Hist windows for geogs')
    dumper(geog_windows, 'geog_windows', step=5)
    geog_sales_class = base_price.get_sales_volume_class(config,
                                           hist_df,
                                           geogs_with_hist,
                                           geog_windows,
                                           len(geogs_with_hist),
                                           params,
                                           flag_sets,
                                           days  = 1 if process_config.isWeeks else 7)
    #logger.info(here + ' Average sales volume calculated')
    dumper(geog_sales_class, 'geog_sales_class', step=6)
    geogs_with_hist = geogs_with_hist.join(geog_sales_class.set_index('geog'), rsuffix  ='_new')
    geogs_with_hist.loc[geogs_with_hist.volume_new.notnull(),'volume'] = geogs_with_hist.loc[geogs_with_hist.volume_new.notnull(),'volume_new']
    geogs_with_hist= geogs_with_hist.drop(['volume_new'], axis =1)
    
    hist_df=manage_seasonality(config,
                               process_config,
                               hist_df,
                               params,
                               geog_sales_class,
                               geogs_with_hist,
                               agg_array,
                               bread)
    #logger.info(here + ' Seasonality applied')
    dumper(hist_df, 'hist_df', step=7)
    hist_df=manage_holidays(config,
                            process_config,
                            hist_df,
                            params,
                            geog_sales_class,
                            geogs_with_hist,
                            agg_array,
                            bread)
    #logger.info(here + ' Holidays applied')
    
    dumper(hist_df, 'hist_df', step=8)
    #price outlier detection
    hist_df, price_result_status = base_price.find_price_outliers(config,
                                  hist_df,
                                  geog_sales_class,
                                  params,
                                  flag_sets,
                                  bread,
                                  process_config)
    #logger.info(here + ' Price Outliers Detected')
    dumper(hist_df, 'hist_df', step=9)
    geogs_with_hist['zeroIsBad'] = exclude_zeros_rule(config, params, geog_sales_class)
    # run main smoothing
    
    overlay_windows = get_overlay_periods(process_config, config, item, )
    dumper(overlay_windows, 'overlay_windows', step=10)
    
    hist_df, result_status = smoothing.create_smoothed_baseline(config,
                                                             hist_df,
                                                             geog_sales_class,
                                                             len(geogs_with_hist),
                                                             geog_windows,
                                                             overlay_windows,
                                                             params,
                                                             process_config,
                                                             flag_sets= flag_sets,
                                                             bread= bread)
    
    #logger.info(here + ' Main smoothing complete')
    dumper(hist_df, 'hist_df', step=11)
    dumper(result_status, 'result_status', step=11)
     #   
    return (    hist_df, 
                result_status,
                geogs_with_hist,
                exception_geogs,
                geog_windows,
                geog_sales_class,
                agg_array)
                

@singledispatch
def get_lookback(config):
    return config.get('max_history_weeks', 0)
@get_lookback.register(Incremental)
def get_lookback_incremental(config, params):
    
    return config.publish_window + config.lookback_weeks


def process_setup(config, process_config, item, hist_end_date, group = P_BASELINE_PARAMS ):
    '''
    Complete build of configuration objects and load the parameters
    
    2016-06-03 -- Adeded logic to respect the hist_start_date on the item.
     
    '''
    #get baseline parameters and update config with all item parameters and attributes.
    config, params = get_item_params(config, item, as_of =hist_end_date, group = In(group))
    
    # Establish the window of hist data we will be processing in thia run
    hist_lookback = get_lookback(config)
    # We can have a hist_start_date controlled by the planner to exlude some days
    
    # We want whole weeks so our hist strt date may be upp to 6 days after 
    # the planner set hist_start_date.
    max_hist_weeks = (hist_end_date - config.hist_start_date).days //7
    if hist_lookback:
        max_hist_weeks = min(max_hist_weeks, hist_lookback)
        
    hist_start_date  = hist_end_date - Week(max_hist_weeks) + Day()
    
    config['hist_start_date'] = hist_start_date  # this overwrites the original value from the item table???
    config['hist_end_date'] = hist_end_date
    config['hist_lookback'] = max_hist_weeks
    
    
    config['publish_start_date'] = hist_end_date - Week(config.publish_window) + Day()
    config['forecast_end_date']  = hist_end_date + Week(process_config.forecast_weeks)
    
    
    if config.publish_start_date  < config.hist_start_date:  
        # we do not construct a baseline  as we do not have enough history
        # everything is going to be satisifed by a proxy
        
        # we trigger the correct result by setting the 
        # hist _start_date to the day afteer the last hist date
        # this will result inan empty hist_df, no geogs_with_hist and no arrays to smooth
        
        config['hist_start_date'] = config.hist_end_date + Day() 
        
        config['run_baseline_processes'] =False
    else:
        config['run_baseline_processes'] =True

    
    
    return config, params

def create_geog_partitions(config, geogs):
    ''' Set up the division of the main work inot partions by geography
    each base geog will be present in exactly one of the partitions.
    '''
    
    base_geog_keys=geogs.geog.values
    
    partition_count = config['partition_count']
    geog_count  = len(base_geog_keys)
    
    # this gives us 
    cuts = [base_geog_keys[(i)*geog_count//partition_count] for i in range (partition_count) ] +[999999]
    return zip(cuts[:-1], cuts[1:])


def map_baseline_on_geog_group(item, 
                               process_config, 
                               config  = None, 
                               cut  = (0,9999999),
                               bread= None, 
                               dumper=None,
                               **kargs):
    
    '''
    This executes as two procedures. They are separated because we expect this to take a long time and 
    we want to be able to rework this as a parallel process.
    The first procedure retrieves history from the database and stores it.
    The second procedure retrieves the stored hist and then proceeds with smoothing.
    
    In a parallel framework we would expect the firsat step to ruin longer and we would run that at a higher degree
    of parallelism.  This procedure would thn be replaced by a dispatch process..
    
    '''
    hist_status, shard_key =  hist_io.build_hist_shard(config, item =item ,geogs  = cut) 
    
    
    #logger.info(bread() + ' Retrieved history ')
    
    return  process_baseline_for_geogs( item, 
                               process_config, 
                               config  = config, 
                               cut  = cut, 
                               bread=bread,
                               dumper=dumper,
                               **kargs) 

@singledispatch
def filter_healthy_baselines(process_config,
                             geogs_with_hist,
                             result_status):  
    return geogs_with_hist.loc[result_status<P_BAD_BASELINE, 'geog']

@filter_healthy_baselines.register(DOW) 
def filter_healthy_baselines_DOW(process_config,
                             geogs_with_hist,
                             result_status): 
    
    
    result_status.columns  = DAYNAMES
    cond =np.logical_or((result_status.values < P_BAD_BASELINE) ,(geogs_with_hist[DAYNAMES].values ==  0)).all(axis = 1)
    # return the good geogs as a series
    return geogs_with_hist.loc[cond,:].reset_index()

def get_holiday_index(config, 
                      process_config, 
                      hist_df, 
                      params, 
                      geogs_with_hist,
                      agg_hierarchy,
                      bread,
                      start='hist_start_date', end='hist_end_date'):
    ''' Collect the sesonality index and apply to each day.
    
  
    '''
    
    
    holiday_idx = apply_seas_holiday_index(config, 
                                      process_config,
                                      'holiday_group', 
                                      holidays.get_holiday_index, 
                                      params,                                 
                                      geogs_with_hist, 
                                      agg_hierarchy, 
                                      bread, 
                                      start = start,
                                      end=end,
                                       )
    
    return (hist_df.assign(holiday_idx = holiday_idx) )

@singledispatch                  
def prepare_results(config, process_config , *args, **kargs):
    raise NotImplementedError(  )

@prepare_results.register(Full)
@prepare_results.register(Incremental)
def prepare_results_normal(config, process_config, item, baseline_df, geogs, geogs_with_hist, results_status, geog_windows,params, agg_array,  bread):
    '''
    '''
    
    
    def process_good_baselines(config, process_config,baseline_df, geogs_with_hist, results_status, geog_windows,params, agg_array,  bread):
        # ad dthe sesonalized baseline on the cleansed deeasonalized baseline
        
        geogs_with_good_baselines = filter_healthy_baselines(process_config,
                                                         geogs_with_hist,
                                                         results_status)
        
        good_baselines = baseline_df.loc[baseline_df.index.get_level_values('geog').isin(geogs_with_good_baselines.geog ),:]
        dumper(good_baselines, 'good_baselines', step=13)
        
        
        # get the final result dfus 
        
        #run the index calculation on the good histories
        # we already have the seasonality index
        good_geog_windows = geog_windows.copy()
        [good_geog_windows.pop(k)  for k in good_geog_windows if k not in set(geogs_with_good_baselines.geog)]
        print 'good_baselines & geog count'
        print len(good_baselines.index.levels[0]), len(geogs_with_good_baselines)
        
        # Potentially move this to opre smoothing
        
 
        # establish the usefulness of adus 
        DOW_factor_good_hist  = smoothing.get_DOW_factors(process_config,
                                                      good_baselines,  
                                                      geogs_with_good_baselines, 
                                                      good_geog_windows, 
                                                      params, 
                                                      config, 
                                                      bread= bread )
        
        return good_baselines, geogs_with_good_baselines, DOW_factor_good_hist
    
    # do this in the data prep?
    baseline_df  = get_holiday_index(config,
                                    process_config,
                                    baseline_df,
                                    params,
                                    geogs_with_history, 
                                    agg_array,
                                    bread)

    flag_sets  = get_flag_sets()
    (good_baselines, 
     geogs_with_good_baselines, 
     DOW_factor_good_hist) = process_good_baselines(config, process_config,baseline_df, geogs_with_hist, results_status, geog_windows,params, agg_array,bread)

    
    # update the volume_class_status on the geogs with hist
    
 
                                                       
    def get_result_dfus(geogs, geogs_with_hist):
        all_open_geogs = geogs.loc[(geogs.open_date<= config.forecast_end_date) &
                                   (geogs.close_date > config.hist_end_date), :]
        
        all_active_participation = get_dfus(item= item,
                                            geog=None,
                                            base_only=True,
                                            start_date = LT(config.forecast_end_date),
                                            end_date = GT(config.hist_end_date) )
                                            
        
        all_active_participation = all_active_participation.loc[all_active_participation.geog.isin(all_open_geogs.geog), :]
        
        cond1  = all_active_participation.geog.isin(geogs_with_hist.index) & geogs_with_hist.volume.notnull()
        cond2 = geogs_with_hist.index.isin(all_active_participation)& geogs_with_hist.volume.notnull()
        
        all_active_participation.loc[cond1, 'volume'] = geogs_with_hist.loc[ cond2, 'volume']
    return all_active_participation

    all_active_participation = get_result_dfus(geogs, geogs_with_hist)
    
    def adu_stuff( config, process_config, good_baselines, geogs_with_good_baselines, agg_array,all_active_participation, flag_sets):
        adus = adu.calc_adus(config, process_config, good_baselines, geogs_with_good_baselines, agg_array, flag_sets, flags ='flags')
        dumper(adus, 'adus', step=14)
        agg_scores = adu.get_adu_score(agg_array.loc[agg_array.geog.isin(all_active_participation.geog),:],
                                   geogs_with_good_baseline.index)
        
        
        
        # apply adus.  This creates a set of "fake"baselines for  dfus that have no 
        adu_proxies = adu.apply_aggregation_level_proxies(config, 
                                                          adus, 
                                                          agg_array.loc[agg_array.geog.isin(all_active_participation.geog ) &
                                                                        ~agg_array.geog.isin(geogs_with_good_baselines),:], 
                                                           adu_scores)
        return adus, adu_proxies
    
    adus, adu_proxies = adu_stuff( config, process_config, good_baselines, geogs_with_good_baselines, agg_array,all_active_participation, flag_sets)
    
    geogs_with_bad_baseline = geogs_with_hist.index.difference(geogs_with_good_baselines.set_index('geog'))
    cond = adu_proxies.get_level_values('geog').isin(geogs_with_bad_baseline )
    adu_proxies.loc[cond,'flags'] = baseline_df[cond, 'flags']

    DOW_factor_no_good_hist  = smoothing.get_DOW_factors(process_config,
                                                      adu_proxies,  
                                                      geogs.loc[geogs.index.isin(adu_procxies.index.get_level_values('geog')), :], 
                                                      geog_windows, 
                                                      config, 
                                                      bread= bread )

    DOW_factor  = pd.concat([DOW_factor_good_hist,DOW_factor_no_good_hist]).sort_index()
    flags = pd.concat(baseline_df.loc[baseline_df.index.get_level_values('bday') >= config.publish_start_date,['flags]']], adu_proxies.loc[:, ['flags']]).sort_index()
    
    def build_composite_index(config, process_config, item, all_geogs, flags, agg_hierachy, flag_sets, DOW_factor):
        # we only publish for the active loactions
        res =  flags.loc[flags.index.get_level_values('geog').isin(all_active_participation.geog)]
        res = manage_seasonality(config, process_config, hist_df, params, 
                              all_geogs, agg_hierarchy, bread,start='publish_start_date', end='hist_forecast_end')
        res = get_holiday_index(config, process_config, res, params, 
                              all_geogs, agg_hierarchy, bread,start='publish_start_date', end='hist_forecast_end')
        res['DOW_index'] =smoothing.explode_DOW_factor(process_config,   all_geogs, DOW_factor, config.publish_window + config.forecast_window ) 
        res['mask'] = anySet(flags.flags, flag_sets[P_HIST_MASK])
        res  = rres.drop(['flags'], axis = 1)
        res = res.rename(columns = dict())
        return res.drop(['flags'], axis = 1)
    
    composite_index= build_composite_index(config, process_config, all_active_participation, flags, agg_hierachy, flag_sets, DOW_factor)
    
    calc_status  = all_active_participation.merge(DOW_factor, left_on='geog', right_index=True, how ='left')
    
    def prepare_baseline_output(hist_df, isADU=False):
        '''
        '''
        hist_df[isSet(hist_df.flags, P_FILL_HIST):[c for c in hist_df if ~hist_df[c].dtype.contains('int')]] = np.nan
        hist_df[isSet(hist_df.flags, P_FILL_HIST):[c for c in hist_df if hist_df[c].dtype.contains('int')]] = 0
        
        return histio.remap_columns(hist_df, isADU)
    baseline_to_store = prepare_baseline_output( baseline_df)
    adus_proxies_to_store = prepare_baseline_output(adu_proxies)
    adus_to_store = prepare_baseline_output( adus, isADUs=True)
    

def main(item,
        config = None,
        IO = None,
        bread = None,
        dumper= dummy_dump):
    ''' 
    
    
    config   -- A switch/ dict that determines the behavior of the program
    '''
   
    # configure sub modules
    exportIO(IO)
    # register this item with the breadcrumb
    bread.push('Item', str(item))
    
    # get demand process details. from customer configuration
    process_config  = get_process_config()
    logging.info('Finish process_config')
    #if #logger.level== logging.DEBUG:
        #logger.debug(pp.pformat(process_config))
        
    hist_end_date = IO.exec_query('business_day').get_top_row().last_post_date
    
    #complete building top level parameter sets
    config, params = process_setup(config, process_config, item, hist_end_date)
    #logger.info('Finish job config')
    #if #logger.level== logging.DEBUG:
        #logger.debug(pp.pformat(config))
        
    #get base_geogs
    geogs, agg_array = collect_geog_data(process_config)
    
    # Complete the build of the parameters by adding the geog information
    params.configure_variable_level(geogs, agg_array)
    # make execution groups
    cuts =  create_geog_partitions(config,geogs )
    # set up the division of the main work inot partions by geography
    # each base geog will be present in exactly one of the 
    base_geog_keys=geogs.geog.values
    
    partition_count = config['partition_count']
    geog_count  = len(base_geog_keys)
    cuts = [base_geog_keys[(i)*geog_count//partition_count] for i in range (partition_count) ] +[999999]
    cuts = zip(cuts[:-1], cuts[1:])
    
    # Get all event details
    # missing parmeter to understand if this item applies all events  -- janan case
    events_df = events.prep_events_for_baseline(item,
                                                    config.hist_start_date,
                                                    config.hist_end_date,
                                                    base_geog_type = process_config.base_geog_type)
    
    
    #logger.info(bread() + 'Retrieved Events')
    # Store events where they can be retrieved in patrallel processes/threads
    event_handle  = events.load_event_store(events_df)
    # Getevents and create the internal representatuion
    
    if config.run_baseline_processes:
        # This is moving too many pieces backwards and forwards
        # 1) Consolidate the geog cardinality objects (except agg_array)into one data frame
        #     but care is =needed if the indexes are not aligned. We may lose the 
        #     understanding of the status of the geog in the context of the original item
        # 2) In parallel execution my trget design is each process will persist its data in a 
        #    local data store and not passed back as a return value
        # 3) Anywaty some of these geog ones areprobably not dataframes at the monment. So when we test thias itr will fail  
        baseline  = [pd.DataFrame()]
        results_status = [pd.DataFrame()]
        geogs_with_hist = [pd.DataFrame()]
        rejected_geogs = [pd.DataFrame()]
        geog_windows = [pd.DataFrame()]
        geog_sales_class = [pd.DataFrame()]
        #agg_array = [pd.DataFrame()]
         
        for i in range( partition_count):
            cut  = cuts[i]
            (res_baseline, 
                    res_results,
                    res_geogs_with_hist,
                    res_rejected_geogs,
                    res_geog_windows,
                    res_geog_sales_class,
                    res_agg_array) = map_baseline_on_geog_group(item ,
                                                                 process_config,
                                                                 config  = config,
                                                                 cut=cut,
                                                                 geogs  = geogs.loc[(geogs.geog >= cut[0]) &
                                                                                     (geogs.geog < cut[1]),:],
                                                                 agg_array =agg_array,
                                                                 params  = params,
                                                                 event_handle = event_handle,
                                                                 bread= bread,
                                                                 dumper = dumper)
            print res_baseline.reset_index().info()
            baseline.append(res_baseline.reset_index())
            results_status.append(res_results.reset_index())
            geogs_with_hist.append(res_geogs_with_hist.reset_index())
            rejected_geogs.append(res_rejected_geogs.reset_index())
            geog_windows.append(res_geog_windows)
            geog_sales_class.append(res_geog_sales_class)
            
       
        baseline_df = pd.concat(baseline, ignore_index=True).set_index(['geog', 'bday']).sort_index()
        results_status = pd.concat(results_status, ignore_index=True).set_index(['geog']).sort_index()
        geogs_with_hist = pd.concat(geogs_with_hist, ignore_index=True).set_index(['geog']).sort_index()
        rejected_geogs = pd.concat(rejected_geogs, ignore_index=True).set_index(['geog']).sort_index()
        
        geog_windows = OrderedDict(sorted(reduce(lambda x, y : x+y,[list(gw.iteritems()) for gw in geog_windows], [])))
                             
        geog_sales_class = pd.concat(geog_sales_class, ignore_index=True).set_index(['geog',]).sort_index()
        #agg_array = pd.concat(agg_array, ignore_index=True)
    else:
        baseline_df = pd.DataFrame()
        geogs_with_hist = pd.DataFrame()
        geog_windows =pd.DataFrame()
        geog_sales_class = pd.DataFrame()
        # aggarry stays with its original value
    # Note agg-array is no longer aligned with geogs_with hist
    # Clean up baselines
    
    
    baseline_df  = cleanse.cleanse_baseline(config, 
                                            process_config,
                                            baseline_df)
    dumper(baseline_df, 'baseline_df', step=12)
    
    return prepare_results(config, process_config, item, baseline_df, geogs_with_hist, results_status, geog_windows,params, agg_array, flag_sets, bread)
    

if __name__ == '__main__':

    pass
    # parse sargs

    # open the database connections
