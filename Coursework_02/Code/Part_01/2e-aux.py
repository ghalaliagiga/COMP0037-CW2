import matplotlib.pyplot as plt

# means and ranges arrays taken from output of q2_e.py
means = [0.22773428, 1.92262618, 2.17430428, 1.69312264, 1.29289636, 0.98949993, 0.77365998, 0.62518373, 0.52110284, 0.45498031, 0.40321304, 0.37303957, 0.34828417, 0.33414211, 0.32137225, 0.29680438, 0.2829174,  0.28019055, 0.27154711, 0.26938762, 0.25995435, 0.25629444, 0.25519056, 0.25493016, 0.25034827, 0.24651994, 0.24013876, 0.24154545, 0.23639744, 0.23931028, 0.23400574, 0.23042083, 0.22555976, 0.22603848, 0.22242423, 0.21865085, 0.21981969, 0.21867369, 0.21756223, 0.21734885]
ranges = [0.8119997978210449, 1.8930010795593262, 1.7329862117767334, 0.957000732421875, 1.3920021057128906, 0.48600006103515625, 0.574986457824707, 0.5910000801086426, -0.3150012493133545, -0.3269989490509033, -0.2979860305786133, -0.28099989891052246, -0.2750124931335449, -0.24700021743774414, -0.24000024795532227, -0.2310020923614502, -0.23654508590698242, -0.22601318359375, -0.2090001106262207, -0.22299981117248535, -0.21057629585266113, -0.18999743461608887, -0.1920003890991211, -0.19800066947937012, -0.20353484153747559, -0.182999849319458, -0.1837010383605957, -0.17500019073486328, -0.18001317977905273, -0.18598675727844238, -0.18399858474731445, -0.18599963188171387, -0.18500113487243652, -0.18199896812438965, -0.18399953842163086, -0.17000222206115723, -0.1640002727508545, -0.17452311515808105, -0.17601251602172852, -0.18198704719543457]

# Plot for means
plt.figure(figsize=(10, 5))
plt.plot(means)
plt.title('Means of Times')
plt.xlabel('Array Index')
plt.ylabel('Times')
plt.savefig('means.png')
plt.show()

# Plot for ranges
plt.figure(figsize=(10, 5))
plt.plot(ranges)
plt.title('Ranges of Times')
plt.xlabel('Array Index')
plt.ylabel('Times')
plt.savefig('ranges.png')
plt.show()