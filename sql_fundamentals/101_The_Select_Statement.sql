select customer_id,count(*) from grocery_db.campaign_data group by customer_id having count(*) > 1 order by customer_id

select distinct campaign_date from grocery_db.campaign_data;

select mailer_type,signup_flag,sum(signup_flag) from grocery_db.campaign_data group by mailer_type,signup_flag;

-----------
select
  * 
from
  grocery_db.product_areas
limit 
  3;
  
select
  * 
from
  grocery_db.customer_details
order by
  distance_from_store, credit_score;
 
---------
-- NTILE - Important one
--------
------------
-- Rounding Data
------------

select
  *,
  round(sales_cost,1) as sales_cost_round1,
  round(sales_cost,0) as sales_cost_round0,
  round(sales_cost,-1) as sales_cost_round_1,
  round(sales_cost,-2) as sales_cost_round_2
from
  grocery_db.transactions
where
  customer_id = 1
  
-----------------

-- Random Sampling

----------------

select 
  * 
from
  grocery_db.customer_details
order by
  random()
limit 100

------------------
--  Extracting parts of date
------------------

select
  distinct transaction_date,
  date_part('day',transaction_date) as day,
  date_part('month',transaction_date) as month,
  date_part('year',transaction_date) as year,
  date_part('dow',transaction_date) as dow
from
  grocery_db.transactions
order by 
  transaction_date;




  








