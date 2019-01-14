import json
import os
import re

filename = 'philmor52_short'

def create_json_from_list(lst):

	"""Create item list in pbtxt format from product list

	lst: list of strings
	"""

	items = []
	# defiine itemID
	itemID = 1
	for i in lst:
		print(json.dumps({'id': itemID, 'name': i}))
		# items.append({'id': itemID, 'name': i})
		items.append(json.dumps({'id': itemID, 'name': i}))
		itemID += 1

	# print('item '.join(items))
	items = '\n'.join(['item {}'.format(i.replace('"', '')) for i in items])
	# print(items)
	return items

def regex_replacement(string):
	"""Insert `"` to class name"""
	# p = re.compile(r'name: (.+)}')
	m = re.sub(r'name: (.+)}', r'name: "\1"}', string)
	return m

def main(lst):
	data = create_json_from_list(lst)

	replaced = regex_replacement(data)

	with open('maps/{}.pbtxt'.format(filename), 'w') as f:
		f.write(replaced)


if __name__ == "__main__":
	main(['BondB#6','BondBS','BondBS25','BondPMix','BondPS','BondRS','BondRS25','BondSCl','BondSCl25','ChfKSB','ChfKSBr','ChfKSR','LMBL','LMLoB','LMLoDSp','LMLoM','LMSmSeR','LMLoSB','LMLoNB','LMFCS','MFM','MG','MR','MS','MTB','MM','MFTLB','MuG','MuP','MuPi','MuVi','NPi','NVi','PaCB','PaCP','PaNtB','PaPB','PaPl','PaSSSl','PaASSl','PaABl','PaSBl','PhMB','PhMB25','PhMG','PhMNB','PhMNM',"PhMNR",'PhMNS','PhMR','PhMR25','PhMRKS'])