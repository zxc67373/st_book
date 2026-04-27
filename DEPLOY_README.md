# SillyTavern 角色卡与世界书部署说明

## 生成信息

- **源文件**: `/home/st_book/novel/1.txt`（22,150 字符）
- **LLM 模型**: `grok-4-1-fast-non-reasoning`
- **API 端点**: `https://api.deerapi.com/v1`
- **生成时间**: 2026-04-10

## 角色卡

共生成 9 个角色卡，SillyTavern V2 格式。

| 角色 | 文件 | 说明 |
|---|---|---|
| 小文 | 小文.json | 出身传统家庭的少年 |
| 陈萍 | 陈萍.json | 小文的母亲 |
| 田莉 | 田莉.json | 邻居女性 |
| 爸爸 | 爸爸.json | 小文的父亲 |
| 年轻男人 | 年轻男人.json | 出场男性角色 |
| 那个女人 | 那个女人.json | 出场女性角色 |
| 李娟 | 李娟.json | 出场女性角色 |
| 王芳 | 王芳.json | 出场女性角色 |
| 沈丽 | 沈丽.json | 出场女性角色 |

**部署路径**: `/home/sillytavern/data/default-user/characters/`

## 世界书

三层架构世界书，SillyTavern V2 格式，共 12 个条目。

| 层级 | 条目数 | 内容 |
|---|---|---|
| 规则层 | 4 | 种族设定、社会规则、经济体系、物理法则 |
| 时间线层 | 1 | 故事总览时间线 |
| 实体层 | 7 | 男孩(小文)、妈妈(陈萍)、爸爸、出轨女人、田莉、年轻男人、网友们 |
| 事件层 | 0 | — |

**部署路径**: `/home/sillytavern/data/default-user/worlds/worldbook_st_v2.json`

## 使用方式

1. 启动 SillyTavern
2. 角色卡：在角色选择界面直接选择对应角色
3. 世界书：在 World Info 界面加载 `worldbook_st_v2`

## 生成流程

```
小说文本 → 文本分块 → LLM提取 → 合并去重 → 筛选 → AI增强角色卡
                                  ↓
                          世界观提取 → 分类 → 三层架构世界书 → ST V2格式转换
```

### 使用的命令

```bash
# 角色卡一键生成
python character_workflow.py auto

# 世界书一键生成
python character_workflow.py wb-auto
```

## 文件结构

```
/home/st_book/
├── cards/                          # 生成的角色卡（9个JSON）
├── worldbook/
│   ├── layered_worldbook.json      # 三层架构世界书（中间产物）
│   └── worldbook_st_v2.json        # SillyTavern V2 格式世界书（最终产物）
├── chunks/                         # 文本分块
├── character_responses/            # 角色提取原始响应
├── roles_json/                     # 合并后的角色档案
├── wb_responses/                   # 世界书提取响应
└── config.yaml                     # 项目配置

/home/sillytavern/data/default-user/
├── characters/                     # 部署的角色卡
└── worlds/                         # 部署的世界书
```
