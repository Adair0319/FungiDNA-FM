# GFF3 文件边缘情况审阅报告

**日期**: 2026-06-09
**源目录**: `/home/lty/yy_projects/fungi_project/1kfg_datasets/annotation_filtered`
**物种目录总数**: 737

---

## 总览

| 类别 | 数量 | 处理策略 |
|------|------|----------|
| A. 同时有 deflines + GeneCatalog | 12 | 按规则：优先 deflines |
| B. 只有 FilteredModels1_deflines | 74 | 直接使用 deflines |
| C. 只有 GeneCatalog | ~509 | 直接使用 GeneCatalog |
| D. 有 FilteredModels1_日期 但无 deflines/GeneCatalog | 25 | **需决策** |
| E. GeneCatalog 有多个版本 | 12 | **需决策** |
| F. 有数字ID前缀的重复文件 | 8 | **需决策** |
| G. 没有任何 .gff3.gz 文件 | 62 | **需决策** |

---

## 类别A: 同时有 FilteredModels1_deflines + GeneCatalog（12个）

> 按规则：优先取 `_FilteredModels1_deflines.gff3.gz`

```
Absrep1
Catan2
Cloaq1
Kocim1
Leucr1
Lobtra1
Micbo1
Prola1
Pseve2
Synrac1
Treen1
Wilmi1
```

---

## 类别B: 只有 FilteredModels1_deflines，无 GeneCatalog（74个）

> 直接取 `_FilteredModels1_deflines.gff3.gz`，无歧义

```
Aaoar1      Agahy1      Amnli1      Antav1      Aplpr1      Aulhe2
Aurvu1      Bacci1      Bysci1      Cadsp1      Clael1      ClaPMI390
Clapy1      ConPMI546   Conth1      Corca1      Cucbe1      Cylto1
Delco1      Denbi1      Didex1      Dotsy1      Elmca1      Exova1
Guyne1      Gymch1      Gymci1_1    Hetpy1      Karrh1      Lenfl1
Leppa1      Leumo1      Lichy1      Linth1      Lopma1      Lopmy1
Macan1      Marpt1      Melpu1      Monpu1      Morco1      Myrdu1
Obbri1      Ophdi1      Panst_KUC8834_1_1  Patat1      Penbr2
Phisc1      Photr1      Phycit1     Pieho1_1    Plesi1      Polci1
Polfu1      Psehy1      Rambr1      Rhivi1      Rhomi1      Ricme1
Sacpr1      Schpa1      Spofi1      Spoli1      Suibr2      Ternu1
ThoPMI491_1 Totfu1      Trepe1      Triol1      Trisp1      Veren1
Wesor1      Xylhe1      Zoprh1
```

---

## 类别C: 只有 GeneCatalog，无 deflines（约509个）

> 直接取唯一的 `_GeneCatalog_*.gff3.gz`，无歧义

<details>
<summary>点击展开完整列表</summary>

```
Abobi1      Abscae1     Acema1      Achstr1     Aciaci1     AcreTS7_1
Acrmed1     Acrvag1     AgarPMI687_1 Agrpra2    Albpec1     Alikh1
Altalt1     Altro1      Altsp012_1  Altsp017_1  Amaapr1     Amarub1
Ampqui1     Amycha1     Amyenc1     Amylap1_1   Amyrou1     Amysub1
Anoalb1     Anobom1     Anokam1     Anomyc1     Ant016_1    Antcit1
Antser1     Aphpse1     Apibac1     Apope1      Aqupat1     Armbor1
Armect1     Armfum1     Armlut1     Armmel1     Armnab1     Armnov1
Armtab1     Artesp1     Auramp1     Baclam1     Basme2finSC Basme92_1
Benpoi1     Bissp1      Blabri1     Blatri1     Blatri_F921_1 Blatri_F986_2
Blyhe1      Boeex1      Bolcoc1     Boledp1     Bolvit1     Bombar1
Bombom1     Borrad1     Botmuc1     Brefa1      Bulin1      Cadosp1
Calful1     Calmar1     Capfu1      CerAGI      Cercau1     Cercer1
Cercru1     Cernew1     Cersc1      Cervir1     Cha1176438  Chabre1
Chafi1      Chafu1      Chagl1      Chahya1     Chame1      Chapip1
Chicu1      Chlpad1     Chocucu1    Chopur1     Chyhya1     Chylag1
Chytri1     Cirumb1     ClaNC0930_1 Clapol1_1   Clarep1_1   Clibor1
Cocst1      CodFL1790_1 Coemoj1     Coespi1     Cokrec1     Conapa1
Conol1      Cont1119283 Copph3      Cornip1     Corsan2     Creces1
Crison1     Crudry1     Crula1      Crypto1     Cryter1     Crywi1
Cunech1     Curina1     Currey1     Cyapal1     Cyaste1     Cylol1
Cysmur1     Cytmel1     Daces1      Dachal1     Dacma1      Dacruf1
Dactor1     Dactort1    Daral1      Daralp1     Darbet1     Darga1
Darpic1     DelFL0756_1 Delst1      Denmi1      Denna1      Diadis1
Diainc1     DiaLGMF1633_1 DiaPMI573_1 Diavoc1   Dicele1     Dicrob1
Didsa1      Digmar1     DimcrSC1    Disdec1     Disorn1     Disven1
Echtin1     Ellano2     Elsamp1     Endsp1      Entech1     Enthel1
Entlig1     Entlut1     Entmai1     Epini1      Epityp1     Exomac1
Favcal1_2   Felpe1      Fenfe1      Fenlin1     FenlinCB54_1 Fibin1
Filflo1     Fimjon1     Fitcyp1     Flafl1      Flaful1     FlaPMI526_1
Fomro1      Fusco1      Fuseq1      Fusoxy1     Fusoxys1    Fusre1
Fusredo2    Fusso1      Fustr1      Fustri1     Fusven1     Gaesem1
Galgeo1     Galinc1     Ganads1     Ganleu1     Gaumor1_1   Geatri1
Gelte1      Gervar1     Gilper1     Glocon1     Glonio1     Glopol1
Gnocas1     Gonbut1     Gorhay1     Gyresc1     Gyrinf1     Halrad1
Helcom1     Helpul1     Helsp1      Hercor1     Herpot1_1   Hesve2finisherSC
Hethyg1     Horac1      Hyacur1     Hyafin1     Hydfim1     Hygaur1
Hygcoc2     Hymvar1     Hypvin2     Hypvoc1     Hyssto1     Idrluna1
Ilyeu1      Ilyro1      Ilyrob1     Inolan1     Intcon1_1   Iscben1
Jahaq1      Jimfl_AD_1  Jimfl_GMNB39_1 Kalbru1  Kalpfe1     Kavalb1
Khuory1     Kicala1     Kircor1     Krezon1     Kurarg1     Lacaka1
Laccon1     Lacdel1     Lachat1     Lachen1     Lacind1     Lacpsa1
Lacpse1     Lacsan1     Lacsu1      Lacsub1     Lactsubd1   Lacviv1
Lacvol1     Lashi1      Lashir1     Lasmin1     Lasov1      Laxbic1
LecAK0013_1 Lenpar1     Lenzyc1     Lepmi1      Lepmol1     Lepor2
Leptod1     Leuca1      Linpe1      Linrh1      Lophiu1     Lopnit1_1
Lopnu1      Loppic1     Lorma1      Lycper1     Macpha1     MarPMI226
Masph1      Matter1     MecolCla_1  Melap1finSC_191 Melbro1  Melen1
MelPMI1271_1 Meltu1     Mersum1     Microd1     Mictri1     Monili2
Moramb1     Morame1     Morana1     Morang1     Morarb1     Morbru1
Morbrun1    Morcon1     Mordel1     Mordim1     Mordis1     Mordun1
Morel_U14_1 Moreoh1     Moresc1     Morexi1     Morfluv1    Morgal1
Morhis1     Morkaki1    Mormul1     Morpal1     Morper1     Morpop1
Morpra1     Morpul1     Morpun1     Morruf1     Morsel1     Morsem1
Morsep1     Morste1     Mortrid1    Morulm1     Morvulg1    MorvulMes17_1
Morwol1     Mrafri1     Mucmuc1     Mucpir1     Muloch1     Myc59_1
Mycafr1     Mycalb1     Mycale1     Mycami1     Mycbel1     Myccro1
Mycden1     Mycepi1     Mycfil1     Mycflo1     Mycgale1    Mychae1
Mycind1     Myclat1     Myclep1     Mycmac1     Mycmet1     Mycoli1
Mycpol1     Mycpur1     Mycreb1     Mycros1     Mycrub1     Mycsan1
Mycvit1     Mycvul1     Myrian1     Myxme1      Necsp1      Neo1551_1
Neobra1     Neora1      Neucr4830_1 Neuhi1      Niavib1     Nidsp1
Obemuc1     Olipa1      OphPMI507_1 Paevar1     Panst_LUM_1_1 Papla1
Parch1      Parchr1     Parmar1     Parpar1     Parsed1     Pauans1
Pcapi1      Pcit11120   Pcit120373  Pcit122482  Pcit122670  Pcit129764
Pcit131864  Pcit141352  Pcit16586   Pcit17464   Pencit1     Perma1
Persub1     Pestal1     Phaart1     PhaeoFL0889_1 Phapla1   PhaPMI808
Phapo1      Phcap1      Phcapi1     Phcit1      Phcitr1     Phefer1
Pheign1_1   Pheni1      Phevit1     Phiasp1     Phimu1      Phlfa1
Phlsub1     Phohig1     Phomu1      Phosp1      Phy27169    Phybla1
Phycap1     Phycapi2    Phycitr1    Phycpc1     Phypa1      Piclef1
Pilano1     Pilbys1     Pilidi1     Pilsph1     Pilumb1     Pincor1_1
Pipcy3_1    Piptie1     Pleav1      Plecto1     Plecuc1     Plecucu2
Plemel1     PlePMI138_1 Plesp1      PnitS607_1  PnitS608_1  PnitS609_1
Podap1      Poddi1      Polagg1_1   Polsie1     Polsqu1     Porchr1
Pormuc1     Pornie1_2   Porspa1     Possti1     Powhir1     Pse1611_1
Pseule1     Ptegra1     Pursp1      Pycful1     Pyrinf1     Pyrly1
Radcon1     Radspe1     Resbic1     Rhanym2_1   Rhesp1      Rhitru1
Rhiund1     Rhivul1     Rhobu1      Ricfib1     Rigmic1     Rorror1
Rusbre1     Ruscom1     Rusdis1     Rusear1     Ruseme1     Ruseme113_1
Rusoch1     Rusrug1     Rusvin1     Sacfar1     Schves1     Sclcihr1
Sclhys1_1   Sclsan1     Sclyun1     Scysp1_1    Serbor1     Sette1
Sidvul1     Simlam1     Sirint1     Sisbri1     Sismus1     Sisrad1
Sisser1     Skebig1     Skeste1     Slopil1     Sorbr1      Sorhu1
Spalat1     Sphbr2      Spifus1     Spogra1_1   Spola1      Spopa1
Spoumb1     Stael1      Stepol1     Stobe1      Sugame1     Suisub1
Synfus1     Synplu1     Synps1      Syzme1      Talpro1     Tercla1
TerNC1134_1 Thacu1      Thaele1     Thasp1      Theglo1     ThyNC0857_1
Tilalb1     Tillet1     Tirniv1     Triarc1     Trigue1     Trihyb1
Tripara1    Tripop1     Truan1      Tubcan1     Tubgib1     Tubmel1
Tubmes1     Tylas1      TyphTRa3160C_1 Umbelo1  Umbisa1     Umbsp_AD052_1
Uniuni1     Uroocc1     Uropr1      Usnflo1     Ustgig1     Varmin1
Velabi1     Vercon1     Verdah1     Wicdo1      Xentul1     Xenvag1
Xerba1      Xylhyp1     XylPMI506   XylPMI703_1 Xylrem1     Zalva1
Zoorad1     Zycmex1     Zyghet1
```
</details>

---

## 类别D: 有 FilteredModels1_YYYY-MM-DD 但无 deflines/GeneCatalog（25个）

> ⚠️ **需你决策**。这些物种没有 `_FilteredModels1_deflines` 也没有 `_GeneCatalog_`，但有带日期的 `_FilteredModels1_YYYY-MM-DD.gff3.gz`（通常还有 `_MitoGenes_`、`_primary/secondary_alleles_` 等伴随文件）

| 物种 | 可用的 .gff3.gz 文件 |
|------|---------------------|
| Cersp5352_1 | FilteredModels1_2024-05-29, FilteredModels1_2024-08-23, primary/sec_alleles (x4), MitoGenes |
| Claher1 | MitoGenes_2025-03-18, FilteredModels1_2025-04-07 |
| Cunbla1 | MitoGenes_2024-04-09, FilteredModels1_2024-04-10 |
| Ememar1 | MitoGenes_2024-04-09, FilteredModels1_2024-04-19 |
| Entmec1 | MitoGenes_2025-08-18, FilteredModels1_2025-08-22 |
| Fempez1 | MitoGenes_2024-08-29, FilteredModels1_2024-09-03, primary/sec_alleles |
| HelPMI846_1 | MitoGenes_2025-03-18, FilteredModels1_2025-04-11 |
| IlyPMI658_1 | FilteredModels1_2025-04-13, MitoGenes_2025-03-18 |
| Lecmu1 | FilteredModels1_2024-05-30（仅一个，无歧义） |
| Niddef1 | MitoGenes_2024-04-19, FilteredModels1_2024-04-22 |
| Sarcolat1 | MitoGenes_2024-07-21, FilteredModels1_2024-07-22 |
| Spiasp1 | MitoGenes_2025-08-18, FilteredModels1_2025-08-22 |
| Tul14233_1 | MitoGenes_2023-11-08, FilteredModels1_2023-11-16, primary/sec_alleles |
| Tulcal1571_1 | FilteredModels1_2025-02-22, MitoGenes_2025-02-05 |
| Tulhel1 | FilteredModels1_2024-05-28, FilteredModels1_2024-08-23, MitoGenes, primary/sec_alleles (x4) |
| Tulirr1 | FilteredModels1_2024-05-28, FilteredModels1_2024-08-22, MitoGenes, primary/sec_alleles (x4) |
| Tulsp217B_1 | FilteredModels1_2024-12-07, FilteredModels1_2024-12-11, MitoGenes, primary/sec_alleles (x4) |
| Tulsp5341_1 | FilteredModels1_2024-12-06, FilteredModels1_2025-01-16, MitoGenes, primary/sec_alleles (x4) |
| Tulsp791A_1 | FilteredModels1_2024-11-22, FilteredModels1_2025-01-24, MitoGenes, primary/sec_alleles (x4) |
| Tulsp791B_1 | FilteredModels1_2024-08-22, FilteredModels1_2024-05-30, MitoGenes, primary/sec_alleles (x4) |
| Tulsp812A_1 | FilteredModels1_2024-12-06, FilteredModels1_2025-01-16, MitoGenes, primary/sec_alleles (x4) |
| TulspSV61_1 | FilteredModels1_2025-03-04, FilteredModels1_2025-02-27, MitoGenes, primary/sec_alleles (x4) |
| Varpro1 | FilteredModels1_2024-07-19, MitoGenes_2024-07-16 |

**问题**：对于这些物种，是否取 `_FilteredModels1_YYYY-MM-DD.gff3.gz`（最新日期的）？是否忽略 `MitoGenes`？

---

## 类别E: GeneCatalog 有多个版本（12个）

> ⚠️ **需你决策**。分为两种子情况：

### E1. 不同日期版本——真正有多个版本（4个）

| 物种 | 可用版本 |
|------|---------|
| AcreTS7_1 | GeneCatalog_20190202, GeneCatalog_20190328 |
| Kalbru1 | GeneCatalog_20210903, GeneCatalog_20210712 |
| Olipa1 | GeneCatalog_20171014, GeneCatalog_20171127 |
| Pcit131864 | GeneCatalog_20200614, GeneCatalog_20200629 |

### E2. 数字ID前缀重复——同一日期同一文件带了不同前缀（8个）

| 物种 | 文件（同一日期，不同前缀） |
|------|--------------------------|
| DimcrSC1 | `7494803-DimcrSC1_GeneCatalog_20151223`, `25641035-DimcrSC1_GeneCatalog_20151223` |
| Entlut1 | `13990292-Entlut1_GeneCatalog_20220126`, `14097765-Entlut1_GeneCatalog_20220126` |
| Gnocas1 | `13990271-Gnocas1_GeneCatalog_20220126`, `14097783-Gnocas1_GeneCatalog_20220126` |
| Krezon1 | `13998171-Krezon1_GeneCatalog_20220127`, `14097736-Krezon1_GeneCatalog_20220127` |
| Lepmol1 | `8859360-Lepmol1_GeneCatalog_20181030`, `8922934-Lepmol1_GeneCatalog_20181030`（还有带前缀的 primary/sec_alleles） |
| Mycafr1 | `4933457-Mycafr1_GeneCatalog_20160506`, `25643341-Mycafr1_GeneCatalog_20160506` |
| Synps1 | `25646337-Synps1_GeneCatalog_20160128`, `7494732-Synps1_GeneCatalog_20160128` |
| Thasp1 | `7494721-Thasp1_GeneCatalog_20160203`, `25646549-Thasp1_GeneCatalog_20160203` |

**问题**：E1 是否取最新日期？E2 是否取任意一个（文件内容应该一样）？

---

## 类别G: 没有任何 .gff3.gz 文件（62个）

> ⚠️ **需你决策**。这些物种目录下只有 `.gff.gz`（不带3）文件

| 物种 | 现有文件 |
|------|---------|
| Acain1 | GeneCatalog_genes_20150309.gff.gz |
| Armga1 | GeneCatalog_genes_20141028.gff.gz |
| Artfe1_2 | GeneCatalog_genes_20150912.gff.gz |
| Ascni1 | GeneCatalog_genes_20141120.gff.gz |
| Caupr1 | 25639279-Caupr1_GeneCatalog_genes_20150107.gff.gz; 7494779-Caupr1_GeneCatalog_genes_20150107.gff.gz |
| Caupr_SCcomb | 25639296-Caupr_SCcomb_GeneCatalog_genes_20140821.gff.gz; 7494767-Caupr_SCcomb_GeneCatalog_genes_20140821.gff.gz |
| Cepal1_1 | GeneCatalog_genes_20150113.gff.gz |
| Cepfr1_1 | GeneCatalog_genes_20150115.gff.gz |
| Cersp1 | GeneCatalog_genes_20150406.gff.gz |
| Chiap1 | GeneCatalog_genes_20150110.gff.gz |
| Cloro1 | GeneCatalog_genes_20150131.gff.gz |
| Cocba1 | GeneCatalog_genes_20140926.gff.gz |
| Copmic2 | GeneCatalog_genes_20150909.gff.gz; FM1_removed_alleles.gff.gz |
| Corma2 | GeneCatalog_genes_20150506.gff.gz |
| Decga1 | GeneCatalog_genes_20150425.gff.gz |
| Erebi1 | GeneCatalog_genes_20150104.gff.gz |
| Eryha1 | GeneCatalog_genes_diploid_20141112.gff.gz; GeneCatalog_genes_20141112.gff.gz |
| Helsul1 | GeneCatalog_genes_20150720.gff.gz |
| Hyabl1 | GeneCatalog_genes_20150410.gff.gz |
| Hydru2 | GeneCatalog_genes_20150110.gff.gz |
| Jamsp1 | GeneCatalog_genes_20141226.gff.gz |
| Lenvul1 | GeneCatalog_genes_20150718.gff.gz |
| Linin1 | GeneCatalog_genes_20141206.gff.gz |
| Lizem1 | GeneCatalog_genes_20150430.gff.gz |
| Lolmi1 | GeneCatalog_genes_20150504.gff.gz |
| Lopni1 | GeneCatalog_genes_20150213.gff.gz |
| Lorju1 | GeneCatalog_genes_20150110.gff.gz |
| Maseb1 | GeneCatalog_genes_20141122.gff.gz |
| Meimi1 | GeneCatalog_genes_20150112.gff.gz |
| Melsp1 | GeneCatalog_genes_20121023.gff.gz |
| Melti1 | GeneCatalog_genes_20150427.gff.gz |
| Metbi_SCcomb | 25643056-Metbi_SCcomb_GeneCatalog_genes_20140820.gff.gz; 7494749-Metbi_SCcomb_GeneCatalog_genes_20140820.gff.gz |
| Micmi1 | GeneCatalog_genes_20150625.gff.gz |
| Mictr1 | GeneCatalog_genes_20140926.gff.gz |
| Monru1 | GeneCatalog_genes_20121211.gff.gz |
| Morimp1 | GeneCatalog_genes_20150807.gff.gz |
| Mutel1 | GeneCatalog_genes_20150215.gff.gz |
| Myrin1 | GeneCatalog_genes_20140928.gff.gz |
| Mytre1 | GeneCatalog_genes_20150115.gff.gz |
| Nieex1 | GeneCatalog_genes_20140925.gff.gz |
| Paxam1 | GeneCatalog_genes_20150425.gff.gz |
| Rhili1 | GeneCatalog_genes_20150422.gff.gz |
| Rhodsp1 | GeneCatalog_genes_20150111.gff.gz |
| Rutfi1 | GeneCatalog_genes_20150501.gff.gz |
| Sarco1 | GeneCatalog_genes_20150213.gff.gz |
| Sepsp1 | GeneCatalog_genes_20150424.gff.gz |
| Setho1 | GeneCatalog_genes_20150424.gff.gz |
| Stagr1 | GeneCatalog_genes_20141015.gff.gz |
| Symat1 | GeneCatalog_genes_20150304.gff.gz; GeneCatalog_genes_20150310.gff.gz |
| Tescy1 | GeneCatalog_genes_20150113.gff.gz |
| Thega1 | GeneCatalog_genes_20150318.gff.gz |
| Themi1 | GeneCatalog_genes_20141205.gff.gz |
| Tilwa1 | GeneCatalog_genes_20141225.gff.gz |
| Torra1 | GeneCatalog_genes_20150428.gff.gz |
| Tribi1 | GeneCatalog_genes_20141117.gff.gz |
| Trich1 | GeneCatalog_genes_20150114.gff.gz |
| Trigu1 | GeneCatalog_genes_20141217.gff.gz |
| Tripe1 | GeneCatalog_genes_20150117.gff.gz |
| Umbra1 | GeneCatalog_genes_20121109.gff.gz |
| Ustsp1 | GeneCatalog_genes_20150106.gff.gz |
| Valla1 | GeneCatalog_genes_20141003.gff.gz |
| Zyghe1_2 | GeneCatalog_genes_20150916.gff.gz |

**问题**：这些 `.gff.gz` 文件是否可用？如果可以，解压后就是 `.gff`（非 `.gff3`），是否接受？

---

## 总结：需要你决策的问题

1. **类别D（25个）**：有 `FilteredModels1_日期` 但无 deflines/GeneCatalog 的，是否取最新日期的 `FilteredModels1_YYYY-MM-DD.gff3.gz`？
2. **类别E1（4个）**：GeneCatalog 有多个日期版本的，是否取最新日期？
3. **类别E2（8个）**：同一文件有不同数字ID前缀的，是否取任意一个？
4. **类别G（62个）**：只有 `.gff.gz` 没有 `.gff3.gz` 的，是否用 `.gff.gz` 代替？
