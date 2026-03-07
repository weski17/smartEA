//+------------------------------------------------------------------+
//|                                            ExportHistoryEA.mq5   |
//|                                                                   |
//| @file    ExportHistoryEA.mq5                                      |
//| @brief   Expert Advisor for MT5 that listens for a Python file   |
//|          trigger and runs the full OHLCV history export.         |
//|                                                                   |
//| @description                                                      |
//|   Monitors a trigger file in the MT5 Common/Files folder.        |
//|   When Python writes "START" to trigger.txt, this EA:            |
//|     1. Exports all timeframes (M1,M15,M30,H1,H4,D1) to CSV      |
//|     2. Writes "DONE" to trigger.txt when finished                |
//|     3. Writes "ERROR" if something went wrong                    |
//|                                                                   |
//| @usage                                                            |
//|   1. Copy to MQL5/Experts/ folder                                |
//|   2. Compile in MetaEditor (F7)                                  |
//|   3. Drag onto any XAUUSD chart                                  |
//|   4. Enable "Allow Algo Trading"                                  |
//|   5. Run Python pipeline — EA responds automatically             |
//+------------------------------------------------------------------+
#property copyright "Custom EA"
#property link      ""
#property version   "1.00"

//--- Input parameters
input string InpSymbol       = "XAUUSD"; // Symbol to export
input int    InpMaxLoadTries = 10;        // Max history load attempts per timeframe
input int    InpLoadWaitMs   = 2000;      // Wait time between load attempts (ms)
input int    InpCheckMs      = 1000;      // How often to check trigger file (ms)

//--- Trigger file name (in Common/Files folder)
#define TRIGGER_FILE "trigger.txt"

//--- Global state
datetime g_lastCheck = 0;
bool     g_isRunning = false;

//+------------------------------------------------------------------+
//| EA initialization                                                |
//+------------------------------------------------------------------+
int OnInit()
  {
   Print("==================================================");
   Print("   ExportHistoryEA MT5 - READY                    ");
   Print("==================================================");
   PrintFormat("[INFO ] Symbol       : %s", InpSymbol);
   PrintFormat("[INFO ] Trigger file : %s (in Common/Files)", TRIGGER_FILE);
   PrintFormat("[INFO ] Check every  : %d ms", InpCheckMs);
   Print("[INFO ] Waiting for Python trigger...");
   Print("--------------------------------------------------");

   //--- Only reset to IDLE if no pending START signal exists
   string current = ReadTrigger();
   if(current != "START")
      WriteTrigger("IDLE");
   else
      Print("[INFO ] START signal already pending — will process on next tick");

   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
//| EA deinitialization                                              |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   WriteTrigger("IDLE");
   Print("[INFO ] ExportHistoryEA stopped");
  }

//+------------------------------------------------------------------+
//| Called on every tick — checks trigger file                       |
//+------------------------------------------------------------------+
void OnTick()
  {
   datetime now = TimeCurrent();
   if((int)(now - g_lastCheck) * 1000 < InpCheckMs)
      return;
   g_lastCheck = now;

   if(g_isRunning)
      return;

   string status = ReadTrigger();
   if(status != "START")
      return;

   g_isRunning = true;
   PrintFormat("[INFO ] === TRIGGER RECEIVED: START @ %s ===",
               TimeToString(now, TIME_DATE | TIME_MINUTES | TIME_SECONDS));

   WriteTrigger("RUNNING");

   bool success = RunExport();

   if(success)
     {
      WriteTrigger("DONE");
      Print("[OK   ] Export complete — trigger set to DONE");
     }
   else
     {
      WriteTrigger("ERROR");
      Print("[ERROR] Export failed — trigger set to ERROR");
     }

   g_isRunning = false;
  }

//+------------------------------------------------------------------+
//| Reads the current status from trigger.txt                        |
//|                                                                  |
//| @return  Status string: START | RUNNING | DONE | ERROR | IDLE   |
//|          or empty string on read failure                         |
//+------------------------------------------------------------------+
string ReadTrigger()
  {
   int handle = FileOpen(TRIGGER_FILE,
                         FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      return "";
   string val = "";
   if(!FileIsEnding(handle))
      val = FileReadString(handle);
   FileClose(handle);
   StringTrimRight(val);
   StringTrimLeft(val);
   return val;
  }

//+------------------------------------------------------------------+
//| Writes a status string to trigger.txt                            |
//|                                                                  |
//| @param  status  The status to write                              |
//+------------------------------------------------------------------+
void WriteTrigger(string status)
  {
   int handle = FileOpen(TRIGGER_FILE,
                         FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(handle == INVALID_HANDLE)
     {
      PrintFormat("[ERROR] Cannot write trigger file: %d", GetLastError());
      return;
     }
   FileWriteString(handle, status);
   FileClose(handle);
  }

//+------------------------------------------------------------------+
//| Forces MT5 to load maximum available history for a timeframe     |
//|                                                                  |
//| @param  symbol  Trading symbol, e.g. "XAUUSD"                   |
//| @param  tf      MT5 ENUM_TIMEFRAMES value, e.g. PERIOD_M15      |
//| @return Number of bars loaded, 0 on failure                      |
//+------------------------------------------------------------------+
int ForceLoadHistory(string symbol, ENUM_TIMEFRAMES tf)
  {
   string pname = PeriodName(tf);
   PrintFormat("[INFO ] Loading history: %s %s", symbol, pname);

   int prevBars = 0, curBars = 0, unchanged = 0;

   for(int attempt = 1; attempt <= InpMaxLoadTries; attempt++)
     {
      //--- MT5: SeriesInfoInteger forces history load
      SeriesInfoInteger(symbol, tf, SERIES_FIRSTDATE);
      Sleep(InpLoadWaitMs);
      curBars = Bars(symbol, tf);
      PrintFormat("[INFO ] Attempt %d/%d: %d bars (prev: %d)",
                  attempt, InpMaxLoadTries, curBars, prevBars);
      if(curBars == prevBars)
        {
         if(++unchanged >= 3)
           { Print("[INFO ] History fully loaded"); break; }
        }
      else unchanged = 0;
      prevBars = curBars;
     }

   if(curBars > 0)
     {
      datetime first[1], last[1];
      CopyTime(symbol, tf, curBars - 1, 1, first);
      CopyTime(symbol, tf, 0,           1, last);
      PrintFormat("[INFO ] Range: %s to %s (%d bars)",
                  TimeToString(first[0], TIME_DATE),
                  TimeToString(last[0],  TIME_DATE),
                  curBars);
     }
   return curBars;
  }

//+------------------------------------------------------------------+
//| Exports one timeframe to CSV in Common/Files folder              |
//|                                                                  |
//| @param  symbol  Trading symbol                                   |
//| @param  tf      MT5 ENUM_TIMEFRAMES value                        |
//| @return True on success, false on failure                        |
//+------------------------------------------------------------------+
bool ExportPeriod(string symbol, ENUM_TIMEFRAMES tf)
  {
   string pname    = PeriodName(tf);
   string filename = symbol + "_" + pname + ".csv";
   datetime tStart = TimeCurrent();

   Print("--------------------------------------------------");
   PrintFormat("[INFO ] Exporting: %s %s -> %s", symbol, pname, filename);

   int bars = ForceLoadHistory(symbol, tf);
   if(bars <= 0)
     {
      PrintFormat("[WARN ] No data for %s %s", symbol, pname);
      return false;
     }

   //--- Copy all OHLCV data at once (MT5 approach)
   MqlRates rates[];
   int copied = CopyRates(symbol, tf, 0, bars, rates);
   if(copied <= 0)
     {
      PrintFormat("[ERROR] CopyRates failed for %s %s: %d", symbol, pname, GetLastError());
      return false;
     }

   int handle = FileOpen(filename,
                         FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON, ';');
   if(handle == INVALID_HANDLE)
     {
      PrintFormat("[ERROR] Cannot open file: %s (error %d)", filename, GetLastError());
      return false;
     }

   //--- Header row
   FileWrite(handle,
             "Datum", "Uhrzeit",
             "Open", "High", "Low", "Close",
             "Volumen", "Session", "Spread_Punkte");

   int digits     = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   int spreadPts  = (int)SymbolInfoInteger(symbol, SYMBOL_SPREAD);
   int step       = (int)MathMax(1, MathRound(copied * 0.10));
   int nextRep    = step;
   int written    = 0;

   //--- Write from oldest to newest
   for(int i = copied - 1; i >= 0; i--)
     {
      datetime barTime = rates[i].time;
      string dateStr   = TimeToString(barTime, TIME_DATE);
      string full      = TimeToString(barTime, TIME_DATE | TIME_MINUTES);
      string timeStr   = StringLen(full) >= 16 ? StringSubstr(full, 11, 5) : "00:00";

      FileWrite(handle,
                dateStr,
                timeStr,
                DoubleToString(rates[i].open,  digits),
                DoubleToString(rates[i].high,  digits),
                DoubleToString(rates[i].low,   digits),
                DoubleToString(rates[i].close, digits),
                (string)rates[i].tick_volume,
                GetSession(barTime),
                (string)spreadPts);
      written++;

      if(written >= nextRep)
        {
         int pct = (int)MathRound(100.0 * written / copied);
         PrintFormat("[INFO ] %s %s: %d/%d bars (%d%%)",
                     symbol, pname, written, copied, pct);
         nextRep += step;
        }
     }

   FileFlush(handle);
   FileClose(handle);

   PrintFormat("[OK   ] %s %s: %d bars written in %ds",
               symbol, pname, written,
               (int)(TimeCurrent() - tStart));
   return true;
  }

//+------------------------------------------------------------------+
//| Runs export for all configured timeframes                        |
//|                                                                  |
//| @return True if at least one timeframe exported successfully     |
//+------------------------------------------------------------------+
bool RunExport()
  {
   datetime tStart = TimeCurrent();
   Print("==================================================");
   Print("   EXPORT START                                   ");
   Print("==================================================");

   ENUM_TIMEFRAMES periods[6] =
     { PERIOD_M1, PERIOD_M15, PERIOD_M30,
       PERIOD_H1, PERIOD_H4,  PERIOD_D1 };

   int    ok = 0, fail = 0;
   string results[6];

   for(int p = 0; p < 6; p++)
     {
      if(ExportPeriod(InpSymbol, periods[p]))
        {
         ok++;
         results[p] = "[OK  ] " + InpSymbol + "_" + PeriodName(periods[p]) + ".csv";
        }
      else
        {
         fail++;
         results[p] = "[FAIL] " + InpSymbol + "_" + PeriodName(periods[p]) + ".csv";
        }
     }

   Print("==================================================");
   Print("   EXPORT SUMMARY                                 ");
   Print("==================================================");
   for(int p = 0; p < 6; p++) Print("   " + results[p]);
   PrintFormat("[INFO ] Success: %d | Failed: %d | Duration: %ds",
               ok, fail, (int)(TimeCurrent() - tStart));
   Print("==================================================");

   return ok > 0;
  }

//+------------------------------------------------------------------+
//| Returns timeframe name string                                    |
//|                                                                  |
//| @param  tf  MT5 ENUM_TIMEFRAMES value                            |
//| @return Timeframe name e.g. "M15", "H1"                         |
//+------------------------------------------------------------------+
string PeriodName(ENUM_TIMEFRAMES tf)
  {
   switch(tf)
     {
      case PERIOD_M1:  return "M1";
      case PERIOD_M15: return "M15";
      case PERIOD_M30: return "M30";
      case PERIOD_H1:  return "H1";
      case PERIOD_H4:  return "H4";
      case PERIOD_D1:  return "D1";
      default:         return "Unknown";
     }
  }

//+------------------------------------------------------------------+
//| Determines trading session from UTC datetime                     |
//|                                                                  |
//| @param  t  Bar open time in UTC                                  |
//| @return Session label: Tokyo | London | NewYork |                |
//|         London+NY | Tokyo+London | Off                           |
//+------------------------------------------------------------------+
string GetSession(datetime t)
  {
   MqlDateTime dt;
   TimeToStruct(t, dt);
   int  mins    = dt.hour * 60 + dt.min;
   bool tokyo   = (mins >= 0   && mins < 540);
   bool london  = (mins >= 420 && mins < 960);
   bool newyork = (mins >= 720 && mins < 1260);
   if(london && newyork) return "London+NY";
   if(tokyo  && london)  return "Tokyo+London";
   if(newyork)           return "NewYork";
   if(london)            return "London";
   if(tokyo)             return "Tokyo";
   return "Off";
  }
//+------------------------------------------------------------------+
