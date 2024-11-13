import xlrd
import json

workbook = xlrd.open_workbook_xls("data.xls")

sheetname = 'data'
worksheet = workbook.sheet_by_name(sheetname)

data = {}

dict = {}


rel_dict = {}


for row_index in range(1 , worksheet.nrows):
    row = worksheet.row_values(row_index)

    # print(row)
    object =row[0]
    object_type=row[1]
    subject=row[2]
    subject_type=row[3]
    predicate=row[4]
    text = row[5]
    # print(text)

    if predicate in rel_dict.keys():
        rel_dict[predicate] += 1
    else:
        rel_dict[predicate] = 1

    if dict.get(text) == None :
        dict[text] = [{"predicate": predicate, "object_type": object_type, "subject_type": subject_type, "object": object, "subject": subject}]
    else :
        dict[text].append({"predicate": predicate, "object_type": object_type, "subject_type": subject_type, "object": object, "subject": subject})

print(rel_dict)

key_list = []
values_list = []

for key in rel_dict.keys():
    key_list.append(key)
    values_list.append(rel_dict[key])

from pyecharts.charts import Bar

bar = Bar()

bar.add_xaxis(key_list)
bar.add_yaxis("Num", values_list)

bar.render("test.html")






# with open('test.json', 'w',encoding='utf8') as outfile:
#
#     for key in dict.keys():
#         # print('{"text":"',key,'","spolist":',dict[key],'}')
#         str1 = '{"text":"' + str(key) + '","spo_list": ' + str(dict.get(key)) + '}' + '\n'
#         outfile.write(str1)


# {"text": "潘惟南，陕西大荔县溢渡村人", "spo_list": [{"predicate": "出生地", "object_type": "地点", "subject_type": "人物", "object": "陕西大荔", "subject": "潘惟南"}]}

# 实体1，实体类型，实体2，实体类型，关系，文本

# for object,object_type,subject,subject_type,predicate,text in enumerate(worksheet.row_values(1,1)) :
#     dict = {}
#     for row in enumerate(worksheet.col_values(0,5), start=2) :
#         triple = worksheet.cell_value()


