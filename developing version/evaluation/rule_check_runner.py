
import pandas as pd


#test 1
succeeded_one_ip_intern = 0
failed_one_ip_intern = 0
def checkOneIPIntern(srcIP,dstIP):
    global succeeded_one_ip_intern
    global failed_one_ip_intern
    if srcIP[:6] == "42.219" or dstIP[:6] == "42.219" or srcIP == "0.0.0.0" or dstIP == "255.255.255.255":
        succeeded_one_ip_intern += 1
    else:
        failed_one_ip_intern += 1

#test 2
succeeded_tcp_80 = 0
failed_tcp_80 = 0
def checkPort80TCP(proto,srcPt,dstPt):
    global succeeded_tcp_80
    global failed_tcp_80
    if srcPt == 80 or srcPt == 443 or dstPt == 80 or dstPt == 443:
        if proto == "TCP":
            succeeded_tcp_80 += 1
        else:
            failed_tcp_80 += 1

#test 3
succeeded_udp_53 = 0
failed_udp_53 = 0
def checkPort53UDP(proto,srcPt,dstPt):
    global succeeded_udp_53
    global failed_udp_53
    if srcPt == 53 or dstPt == 53:
        if proto == "UDP":
            succeeded_udp_53 += 1
        else:
            failed_udp_53 += 1

#test 4
succeeded_multicast = 0
failed_multicast = 0
def checkMultiBroadcast(srcIP,dstIP,row):
    global succeeded_multicast
    global failed_multicast
    ip1_1 = int( srcIP.split(".")[0] )
    ip1_4 = int( srcIP.split(".")[3] )

    ip2_1 = int( dstIP.split(".")[0] )
    ip2_4 = int( dstIP.split(".")[3] )

    if (ip2_1 > 223 or (ip2_1 == 192 and ip2_4 == 255)) and ip1_1 < 224 and not(ip1_4 == 192 and ip1_4 == 255):
        succeeded_multicast += 1
    elif ip1_1 > 223 or (ip1_4 == 192 and ip1_4 == 255):
        failed_multicast += 1

#test5
succeeded_netbios = 0
failed_netbios = 0
def checkNetbios(srcIP,dstIP,dstPt,proto):
    global succeeded_netbios
    global failed_netbios
    ip1_1 = int( srcIP.split(".")[0] )
    ip1_2 = int( srcIP.split(".")[1] )

    ip2_1 = int( dstIP.split(".")[0] )
    ip2_4 = int( dstIP.split(".")[3] )

    if dstPt == 137 or dstPt == 138:
        if ip1_1 == 42 and ip1_2 == 219 and proto == "UDP" and ip2_1 == 42 and ip2_4 == 255:
            succeeded_netbios += 1
        else:
            failed_netbios += 1

#test6
succeeded_byte_packet = 0
failed_byte_packet = 0
def checkRelationBytePackets(bytzes,packets):
    global succeeded_byte_packet
    global failed_byte_packet

    if bytzes >= packets * 42 and bytzes <= packets * 65536:
        succeeded_byte_packet += 1
    else:
        failed_byte_packet += 1

#test7
succeeded_dur_one_packet = 0
failed_dur_one_packet = 0
def checkDurationOnePacket(duration,packets):
    global succeeded_dur_one_packet
    global failed_dur_one_packet
    if packets == 1:
        d = float(duration)
        if d < 1: # duration == "0.000" or duration == "0" or d == 0: 
            succeeded_dur_one_packet += 1
        else:
            failed_dur_one_packet += 1

def output_rst(data_name, test_name, true_count, false_count):
    tot_count = true_count + false_count
    with open('results/rule_check_results.txt','a') as f:
        #print(data_name, '='*10, file=f)
        if tot_count > 0:
            print(test_name ,'true', true_count,'total', tot_count,'percent', true_count/tot_count, file=f)
        else:
            print(test_name, 'no sample', file=f)

def test_one_piece(data_idx, piece_idx):
    files = ['arcnn_f90', 'wpgan', 'ctgan', 'bsl1', 'bsl2', 'real'] 
    if files[data_idx] == 'real':
        df = pd.read_csv("./postprocessed_data/%s/day2_90user.csv" % files[data_idx])
    else:
        df = pd.read_csv("./postprocessed_data/%s/%s_piece%d.csv" % (files[data_idx], files[data_idx], piece_idx))

    for index, row in df.iterrows():
        #print(row['c1'], row['c2'])
        checkOneIPIntern(row['sa'], row['da'])
        checkPort80TCP(row['pr'],row['sp'],row['dp'])
        checkPort53UDP(row['pr'],row['sp'],row['dp'])
        #checkNetbios(row['sa'], row['da'],row['dp'],row['pr'])
        checkRelationBytePackets(row['byt'],row['pkt'])
        checkDurationOnePacket(row['td'],row['pkt'])
 
    data_name = '%s_piece%d' % (files[data_idx], piece_idx)
    with open('results/rule_check_results.txt','a') as f:
        print(data_name, '='*10, file=f)
    output_rst(data_name, 'test1', succeeded_one_ip_intern, failed_one_ip_intern)
    output_rst(data_name, 'test2', succeeded_tcp_80, failed_tcp_80)
    output_rst(data_name, 'test3', succeeded_udp_53, failed_udp_53)
    #output_rst(data_name, 'test4', succeeded_netbios, failed_netbios)
    output_rst(data_name, 'test5', succeeded_byte_packet, failed_byte_packet)
    output_rst(data_name, 'test6', succeeded_dur_one_packet, failed_dur_one_packet)
    #output_rst('test7',

if __name__ == "__main__":
    for i in range(5, 6):
        for j in range(5):
            test_one_piece(i,j)
