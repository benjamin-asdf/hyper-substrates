(ns animismic.lib.blerp
  (:require
    [animismic.lib.particles :as p]
    [bennischwerdtner.pyutils :as pyutils :refer
     [*torch-device*]]
    [bennischwerdtner.hd.binary-sparse-segmented :as hd]
    [tech.v3.datatype.functional :as f]
    [tech.v3.tensor :as dtt]
    [fastmath.random :as fm.rand]
    [fastmath.core :as fm]
    [libpython-clj2.require :refer [require-python]]
    [libpython-clj2.python :refer [py. py..] :as py]
    [tech.v3.datatype.unary-pred :as unary-pred]
    [tech.v3.datatype.argops :as dtype-argops]
    [bennischwerdtner.sdm.sdm :as sdm]
    [bennischwerdtner.hd.codebook-item-memory :as codebook]
    [bennischwerdtner.hd.ui.audio :as audio]
    [bennischwerdtner.hd.data :as hdd]))

(def blerp-map
  (hdd/clj->vsa* {:heliotrope 1 :orange (hd/->seed)}))

(defn blerp-resonator-force
  [particle-id]
  (hdd/clj->vsa* [:?= 1 [:. blerp-map particle-id]]))


(comment
  (berp-resonator-force :heliotrope)
  (berp-resonator-force :heliotrope))
