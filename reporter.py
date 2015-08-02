import Globals
class Reporter:
    msg = ''
    level = 1


    def __init__(self, level = 1):
        self.level = level

    def new_report(this, string):
        msg = '\n'
        msg += '='*30
        msg += string
        msg += '='*30
        
        if this.level>0:
            print(msg)
        
        this.msg +=  msg
        
    def eraseReport(this):
        this.msg = ''

    def subreport(this, title):
        msg = '\n'        
        msg += '-'*40
        msg += title
        msg += '-'*20
        
        if this.level>0:
            print(msg)
        
        this.msg += msg
        
        
    def __init__(this, report_name = None, debug_level=1):
        this.level = debug_level
        if report_name != None:
            this.new_report(report_name)
        
    def report(this, string):
        if this.level>0:
            print(string)
            
        this.msg += '\n' + string
        
    def concat_report(this, report):
        try:        
            this.msg += '\n' + report.getReport()
        except:
            this.msg += '\n' + report
            
    def getReport(this):
        return this.msg
        
    def saveReport(this, path, filename):
        #make path and file first
        Globals.mkdir_LR(path)
        with open(path+filename, "w") as text_file:
            print(this.getReport(), file=text_file)