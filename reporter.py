class Reporter:
    msg = ''
    level = 1
    
    def __init__(this, debug_level=1):
        this.level = debug_level
        
    def report(this, string):
        if this.level>0:
            print(string)
            
        this.msg += string + "   ||   "
        
    def new_report(this, string):
        msg = ''
        msg += '-'*40
        msg += string
        msg += '-'*20
        
        if this.level>0:
            print(msg)
        
        this.msg += '/n' + msg
    
    def getReport(this):
        return this.msg
        
        