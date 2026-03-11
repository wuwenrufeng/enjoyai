-- 创建游戏数据分析数据库
CREATE DATABASE IF NOT EXISTS game_analytics;

-- 创建测试数据库
CREATE DATABASE IF NOT EXISTS test_db;

-- 用户档案表
CREATE TABLE IF NOT EXISTS game_analytics.user_profiles
(
    user_id UInt64 COMMENT '用户ID',
    register_date Date COMMENT '注册日期',
    country String COMMENT '国家',
    device_type String COMMENT '设备类型',
    channel String COMMENT '渠道来源',
    level UInt16 COMMENT '当前等级',
    vip_level UInt8 COMMENT 'VIP等级',
    last_login_time DateTime COMMENT '最后登录时间'
) ENGINE = MergeTree()
ORDER BY user_id
COMMENT '用户档案表';

-- 用户行为事件表
CREATE TABLE IF NOT EXISTS game_analytics.user_events
(
    event_time DateTime COMMENT '事件时间',
    user_id UInt64 COMMENT '用户ID',
    event_type String COMMENT '事件类型',
    event_name String COMMENT '事件名称',
    level UInt16 COMMENT '用户等级',
    duration UInt32 COMMENT '持续时长(秒)',
    properties String COMMENT '事件属性(JSON)'
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(event_time)
ORDER BY (event_time, user_id)
COMMENT '用户行为事件表';

-- 付费记录表
CREATE TABLE IF NOT EXISTS game_analytics.payment_records
(
    payment_time DateTime COMMENT '付费时间',
    user_id UInt64 COMMENT '用户ID',
    order_id String COMMENT '订单ID',
    amount Decimal(10, 2) COMMENT '付费金额',
    currency String COMMENT '货币类型',
    product_id String COMMENT '商品ID',
    product_name String COMMENT '商品名称',
    payment_channel String COMMENT '支付渠道'
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(payment_time)
ORDER BY (payment_time, user_id)
COMMENT '付费记录表';

-- 每日统计表
CREATE TABLE IF NOT EXISTS game_analytics.daily_stats
(
    stat_date Date COMMENT '统计日期',
    dau UInt32 COMMENT '日活跃用户数',
    new_users UInt32 COMMENT '新增用户数',
    revenue Decimal(12, 2) COMMENT '收入',
    arpu Decimal(8, 2) COMMENT '平均每用户收入',
    retention_day1 Float32 COMMENT '次日留存率',
    retention_day7 Float32 COMMENT '7日留存率'
) ENGINE = MergeTree()
ORDER BY stat_date
COMMENT '每日统计表';

-- 插入测试数据：用户档案
INSERT INTO game_analytics.user_profiles VALUES
(1001, '2024-01-01', 'CN', 'iOS', 'facebook', 25, 3, '2024-01-20 18:30:00'),
(1002, '2024-01-02', 'US', 'Android', 'google', 18, 0, '2024-01-21 12:15:00'),
(1003, '2024-01-02', 'CN', 'iOS', 'tiktok', 32, 5, '2024-01-22 20:45:00'),
(1004, '2024-01-03', 'JP', 'Android', 'facebook', 15, 2, '2024-01-22 09:20:00'),
(1005, '2024-01-05', 'CN', 'iOS', 'organic', 28, 4, '2024-01-23 14:10:00');

-- 插入测试数据：用户行为事件
INSERT INTO game_analytics.user_events VALUES
('2024-01-20 10:00:00', 1001, 'login', 'user_login', 25, 0, '{}'),
('2024-01-20 10:05:00', 1001, 'level_up', 'level_complete', 25, 300, '{"from_level":24,"to_level":25}'),
('2024-01-20 10:30:00', 1001, 'battle', 'pvp_match', 25, 480, '{"result":"win","opponent_id":1003}'),
('2024-01-21 12:00:00', 1002, 'login', 'user_login', 18, 0, '{}'),
('2024-01-21 12:15:00', 1002, 'quest', 'daily_quest_complete', 18, 120, '{"quest_id":"daily_001"}'),
('2024-01-22 20:30:00', 1003, 'login', 'user_login', 32, 0, '{}'),
('2024-01-22 20:45:00', 1003, 'battle', 'pvp_match', 32, 420, '{"result":"loss","opponent_id":1001}'),
('2024-01-22 09:00:00', 1004, 'login', 'user_login', 15, 0, '{}'),
('2024-01-23 14:00:00', 1005, 'login', 'user_login', 28, 0, '{}'),
('2024-01-23 14:20:00', 1005, 'purchase', 'shop_visit', 28, 60, '{"shop_type":"premium"}');

-- 插入测试数据：付费记录
INSERT INTO game_analytics.payment_records VALUES
('2024-01-15 14:30:00', 1001, 'ORD20240115001', 6.99, 'USD', 'vip_monthly', 'VIP月卡', 'apple'),
('2024-01-18 16:20:00', 1003, 'ORD20240118001', 99.99, 'USD', 'gem_pack_large', '大额宝石礼包', 'apple'),
('2024-01-20 11:00:00', 1001, 'ORD20240120001', 0.99, 'USD', 'coin_pack_small', '金币小礼包', 'apple'),
('2024-01-22 15:30:00', 1005, 'ORD20240122001', 19.99, 'USD', 'level_pack_30', '30级礼包', 'apple'),
('2024-01-22 21:00:00', 1003, 'ORD20240122002', 49.99, 'USD', 'gem_pack_medium', '中额宝石礼包', 'apple');

-- 插入测试数据：每日统计
INSERT INTO game_analytics.daily_stats VALUES
('2024-01-20', 1250, 150, 1580.50, 1.26, 0.45, 0.28),
('2024-01-21', 1320, 180, 1850.20, 1.40, 0.48, 0.30),
('2024-01-22', 1420, 200, 2150.80, 1.51, 0.52, 0.32),
('2024-01-23', 1380, 165, 1920.40, 1.39, 0.50, 0.31);

-- 创建一些测试表（用于测试 list_tables 等功能）
CREATE TABLE IF NOT EXISTS test_db.test_table_1
(
    id UInt32,
    name String,
    created_at DateTime
) ENGINE = MergeTree()
ORDER BY id;

CREATE TABLE IF NOT EXISTS test_db.test_table_2
(
    user_id UInt64,
    score Int32,
    timestamp DateTime
) ENGINE = MergeTree()
ORDER BY user_id;
