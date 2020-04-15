import glob
import os
import time
import numpy as np
import cv2
from skimage.measure import compare_ssim

input_dir_sol1 = './results/test_residual_4000/final/'
input_dir_sol2 = './results/test_validation_and_batch_4000/final/'

sol1_name = ''.join(input_dir_sol1.split('/')[2].split('_')[1])     # to get to the name of the tested solution
sol2_name = ''.join(input_dir_sol2.split('/')[2].split('_')[1])

result_dir = "./results/evaluation/" + time.strftime('%Y_%m_%d') + "_SSIM_" + sol1_name + "_vs_" + sol2_name + ".txt"


def calculate_ssim_for_ratio(list, fns_gt, fns_out):
    for index in range(len(fns_gt)):
        gt_image = cv2.imread(fns_gt[index])
        out_image = cv2.imread(fns_out[index])

        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
        out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2GRAY)

        list.append(compare_ssim(gt_image, out_image))


# create lists to store SSIM for ratios
sol1_100 = []
sol1_250 = []
sol1_300 = []

sol2_100 = []
sol2_250 = []
sol2_300 = []

# first calculate SSIM for solution 1

# load gt and output pairs for respected ratios
starting_time = time.time()
ratio_100_gt = glob.glob(input_dir_sol1 + '*100_gt.png')
ratio_100_out = glob.glob(input_dir_sol1 + '*100_out.png')
ratio_250_gt = glob.glob(input_dir_sol1 + '*250_gt.png')
ratio_250_out = glob.glob(input_dir_sol1 + '*250_out.png')
ratio_300_gt = glob.glob(input_dir_sol1 + '*300_gt.png')
ratio_300_out = glob.glob(input_dir_sol1 + '*300_out.png')
print("Finished loading filenames.   Time=%.3f" % (time.time() - starting_time))

starting_time = time.time()
calculate_ssim_for_ratio(sol1_100, ratio_100_gt, ratio_100_out)
print("Finished calculating SSIM for ratio 100.   Time=%.3f" % (time.time() - starting_time))

starting_time = time.time()
calculate_ssim_for_ratio(sol1_250, ratio_250_gt, ratio_250_out)
print("Finished calculating SSIM for ratio 250.   Time=%.3f" % (time.time() - starting_time))

starting_time = time.time()
calculate_ssim_for_ratio(sol1_300, ratio_300_gt, ratio_300_out)
print("Finished calculating SSIM for ratio 300.   Time=%.3f" % (time.time() - starting_time))

average_sol1_100 = np.mean(sol1_100)
average_sol1_250 = np.mean(sol1_250)
average_sol1_300 = np.mean(sol1_300)
average_sol1_whole = np.mean(sol1_100 + sol1_250 + sol1_300)

print("Number of photos in ratio 100 - gt, out:", len(ratio_100_gt), len(ratio_100_out))
print("Number of photos in ratio 250 - gt, out:", len(ratio_250_gt), len(ratio_250_out))
print("Number of photos in ratio 300 - gt, out:", len(ratio_300_gt), len(ratio_300_out))

print("Average", sol1_name, "100:", average_sol1_100)
print("Average", sol1_name, "250:", average_sol1_250)
print("Average", sol1_name, "300:", average_sol1_300)
print("Average", sol1_name, "whole:", average_sol1_whole)

print()

# calculate SSIM for solution 2
starting_time = time.time()
ratio_100_gt = glob.glob(input_dir_sol2 + '*100_gt.png')
ratio_100_out = glob.glob(input_dir_sol2 + '*100_out.png')
ratio_250_gt = glob.glob(input_dir_sol2 + '*250_gt.png')
ratio_250_out = glob.glob(input_dir_sol2 + '*250_out.png')
ratio_300_gt = glob.glob(input_dir_sol2 + '*300_gt.png')
ratio_300_out = glob.glob(input_dir_sol2 + '*300_out.png')
print("Finished loading filenames.   Time=%.3f" % (time.time() - starting_time))

starting_time = time.time()
calculate_ssim_for_ratio(sol2_100, ratio_100_gt, ratio_100_out)
print("Finished calculating SSIM for ratio 100.   Time=%.3f" % (time.time() - starting_time))

starting_time = time.time()
calculate_ssim_for_ratio(sol2_250, ratio_250_gt, ratio_250_out)
print("Finished calculating SSIM for ratio 250.   Time=%.3f" % (time.time() - starting_time))

starting_time = time.time()
calculate_ssim_for_ratio(sol2_300, ratio_300_gt, ratio_300_out)
print("Finished calculating SSIM for ratio 300.   Time=%.3f" % (time.time() - starting_time))

average_sol2_100 = np.mean(sol2_100)
average_sol2_250 = np.mean(sol2_250)
average_sol2_300 = np.mean(sol2_300)
average_sol2_whole = np.mean(sol2_100 + sol2_250 + sol2_300)

print("Number of photos in ratio 100 - gt, out:", len(ratio_100_gt), len(ratio_100_out))
print("Number of photos in ratio 250 - gt, out:", len(ratio_250_gt), len(ratio_250_out))
print("Number of photos in ratio 300 - gt, out:", len(ratio_300_gt), len(ratio_300_out))

print("Average", sol2_name, "100:", average_sol2_100)
print("Average", sol2_name, "250:", average_sol2_250)
print("Average", sol2_name, "300:", average_sol2_300)
print("Average", sol2_name, "whole:", average_sol2_whole)

# output to file
sol1_better_than_sol2 = []
sol2_better_than_sol1 = []
sol1_equal_sol2 = []
total_sol1_evaluations = len(sol1_100) + len(sol1_250) + len(sol1_300)
total_sol2_evaluations = len(sol2_100) + len(sol2_250) + len(sol2_300)

with open(result_dir, "w") as ssim_log:
    print("Full solution1 path:", input_dir_sol1, "\t", "Full solution2 path:", input_dir_sol2, file=ssim_log)
    print(file=ssim_log)

    print("Average", sol1_name, "100:", average_sol1_100, "\t", "Average", sol2_name, "100:", average_sol2_100,
          file=ssim_log)
    print("Average", sol1_name, "250:", average_sol1_250, "\t", "Average", sol2_name, "250:", average_sol2_250,
          file=ssim_log)
    print("Average", sol1_name, "300:", average_sol1_300, "\t", "Average", sol2_name, "300:", average_sol2_300,
          file=ssim_log)
    print("Average", sol1_name, "whole:", average_sol1_whole, "\t", "Average", sol2_name, "whole:", average_sol2_whole,
          file=ssim_log)
    print(file=ssim_log)

    print("Number of evaluated files:", "\t", sol1_name, total_sol1_evaluations, "\t", sol2_name, total_sol2_evaluations, file=ssim_log)
    print(file=ssim_log)

    for index in range(len(ratio_100_gt)):
        string_output = f"{os.path.basename(ratio_100_gt[index])} \t {sol1_name} ssim: {sol1_100[index]} \t {sol2_name} ssim: {sol2_100[index]}"

        if sol1_100[index] > sol2_100[index]:
            sol1_better_than_sol2.append(string_output)
        elif sol1_100[index] < sol2_100[index]:
            sol2_better_than_sol1.append(string_output)
        else:
            sol1_equal_sol2.append(string_output)

        print(string_output, file=ssim_log)

    for index in range(len(ratio_250_gt)):
        string_output = f"{os.path.basename(ratio_250_gt[index])} \t {sol1_name} ssim: {sol1_250[index]} \t {sol2_name} ssim: {sol2_250[index]}"

        if sol1_250[index] > sol2_250[index]:
            sol1_better_than_sol2.append(string_output)
        elif sol1_250[index] < sol2_250[index]:
            sol2_better_than_sol1.append(string_output)
        else:
            sol1_equal_sol2.append(string_output)

        print(string_output, file=ssim_log)

    for index in range(len(ratio_300_gt)):
        string_output = f"{os.path.basename(ratio_300_gt[index])} \t {sol1_name} ssim: {sol1_300[index]} \t {sol2_name} ssim: {sol2_300[index]}"

        if sol1_300[index] > sol2_300[index]:
            sol1_better_than_sol2.append(string_output)
        elif sol1_300[index] > sol2_300[index]:
            sol2_better_than_sol1.append(string_output)
        else:
            sol1_equal_sol2.append(string_output)

        print(string_output, file=ssim_log)

    print(file=ssim_log)

    print(sol1_name, "solution performed better than", sol2_name, "on", len(sol1_better_than_sol2), " out of", total_sol1_evaluations,
          "files, representing", len(sol1_better_than_sol2) * total_sol1_evaluations / 100, "percent.", file=ssim_log)
    print("The files are:", file=ssim_log)
    print(*sol1_better_than_sol2, sep="\n", file=ssim_log)

    print(file=ssim_log)

    print(sol2_name, "solution performed better than", sol1_name, "on", len(sol2_better_than_sol1), " out of", total_sol2_evaluations,
          "files, representing", len(sol2_better_than_sol1) * total_sol2_evaluations / 100, "percent.", file=ssim_log)
    print("The files are:", file=ssim_log)
    print(*sol2_better_than_sol1, sep="\n", file=ssim_log)

    print(file=ssim_log)

    print(sol1_name, "solution performed same as", sol2_name, "on", len(sol1_equal_sol2), " out of", total_sol1_evaluations,
          "files, representing", len(sol1_equal_sol2) * total_sol1_evaluations / 100, "percent.", file=ssim_log)
    print("The files are:", file=ssim_log)
    print(*sol1_equal_sol2, sep="\n", file=ssim_log)


# spravit si array input ciest ako v pri trenovani
# nasledne v loope pocitat jednotlive SSIM
# tie ukladat do pola
# na zaver vypocitat priemer
# pri pocitani brat ohlad na RATIO!
# vypisat aj porovnanie jednotlivych SSIM - nazov fotky / pipiline orig / SSIM / pipeline mine / SSIM
# highlightnut pripady kedy moj vysledok mal lepsi pnsr

# TODO pridat aj kalkulaciu SSIM PRED pridanim rezidualnych blokov? Kedze som upravoval
# sposob nacitavania tak to mohlo zmenit aj samotne vysledky a teda neviem ci
# 1) uprava nacitania zlepsila/zhorsila vysledky
# 2) residualne bloky vylepsili to co sa mohlo stratit upravou nacitania alebo ci mali vobec nejaky ucinok

# pridat naming podla suboru nech mozem pekne pozit aj na ine implementacie
# refaktorovat na viac genericke solution1 and solution2, kde potom iba povedat ze sol1 je residual, sol2 je orig...
# pridat kde su rovnake
