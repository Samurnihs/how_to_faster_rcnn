import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import argparse


def xml_to_csv(path): # parses xml files and adds results to Pandas DataFrame 
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def process_xml(inp, out, out_label):
    xml_df = pd.DataFrame()
    for fold in inp:
    	xml_df = pd.concat([xml_df, xml_to_csv(fold)])
    labels = xml_df['class'].unique() # list of classes
    labels.sort()
    
    file = open(out_label, 'w') # file for labelmap
    for i in range(len(labels)): 
    	file.write(str(labels[i])+'\t'+str(i+1)+'\n')
    file.close()
 	
    xml_df.to_csv(out, index=None)


if __name__ =='__main__':
    parser = argparse.ArgumentParser( #parsing arguments
    description='Turns xml files to csv.')
    parser.add_argument('input_folder', type=str, nargs='+', help='Path to folder with xml files.')
    parser.add_argument('-out', type=str, help='Path to output csv file.', default=os.path.join(os.getcwd(), 'labels.csv'))
    parser.add_argument('-lb', type=str, help='Path to output labelmap txt file.', default=os.path.join(os.getcwd(), 'labelmap.txt'))
    args = parser.parse_args()
    
    	
    process_xml(args.input_folder, args.out, args.lb)

