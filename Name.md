下面给你一些更短但仍清晰的命名备选（按“短 → 信息更全”分组），你可以挑一个风格统一到仓库里：
demo_decode_novelty_check_eform_m3gnet

1) 最短、最像“给用户跑一下”的

demo_dn/（dn = decode + novelty）

dn_demo/

quicktest_dn/

smoke_dn/（smoke test 风格）

2) 清晰直白、不用缩写

demo_decode_novelty/

decode_novelty_demo/

demo_decode_novelty_eform/（如果 eform 是这个 demo 的核心输出之一）

3) 保留“模型/方法”信息，但仍比现在短

demo_dn_m3gnet/

demo_decode_novelty_m3gnet/

demo_dn_eform_m3gnet/（基本等价你现在表达，但短一截）

demo_dn_eform/（不绑定具体 relax 模型名）

4) 和你们现有 2_decode、3_novelty 体系更“对齐”的

2_3_demo/

23_demo_dn/

2d3n_demo/（更短但略“黑话”）

2_decode_3_novelty_demo/（最对齐，但比上面长一点）





















方案 A（保留顺序、突出差异）

01_eform_end2end_oneclick.sh（一体化 demo，一键跑通）

02_eform_end2end_twostep.sh（两步：decode → novelty）

方案 B（更“教程化”，适合放 README）

01_quickstart_eform_oneclick.sh

02_tutorial_eform_decode_novelty_twostep.sh

