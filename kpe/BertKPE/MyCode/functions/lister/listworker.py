

def sort_list(data_list, index, reverse=True):
    """ index: int number, according to it to sort data_list"""
    sorted_data_list = sorted(data_list, key=lambda x: x[index], reverse=reverse)
    return sorted_data_list


def find_answer_positions(document, answers):
    ''' 
    merge the same answers & keep present answers in document
    Inputs:
        document : a word list : ['sun', 'sunshine', ...] || lower cased
        answers : can have more than one answer : [['sun'], ['key','phrase'], ['sunshine']] || not duplicate
    Outputs:
        all_present_answers : prensent answers
        positions_for_all : start_end_posisiton for prensent answers
        a present answer postions list : every present's positions in documents, 
        each answer can be presented in several postions .
        [[[0,0],[20,21]], [[1,1]]]
    '''
    tot_doc_char = ' '.join(document)
    
    positions_for_all = []
    all_present_answers = []
    for answer in answers:
        ans_string = ' '.join(answer)
        
        if ans_string not in tot_doc_char:
            continue
        else: 
            positions_for_each = []
            # find all positions for each answer
            for i in range(0, len(document) - len(answer) + 1):
                Flag = False
                if answer == document[i:i+len(answer)]:
                    Flag = True
                if Flag:
                    assert (i+len(answer)-1) >= i
                    positions_for_each.append([i, i+len(answer)-1])
        if len(positions_for_each) > 0 :
            positions_for_all.append(positions_for_each)
            all_present_answers.append(answer)
            
    assert len(positions_for_all) == len(all_present_answers)
    
    if len(all_present_answers) == 0:
        return None
    
    return all_present_answers, positions_for_all
