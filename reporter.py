class Reporter:
    msg = ''
    level = 1

    def new_report(this, string):
        msg = '\n'
        msg += '='*30
        msg += string
        msg += '='*30
        
        if this.level>0:
            print(msg)
        
        this.msg +=  msg

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
        
    def saveReport(this, path):
        with open(path, "w") as text_file:
            print(this.getReport(), file=text_file)