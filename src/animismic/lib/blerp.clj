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

;; (def blerp-map
;;   (hdd/clj->vsa* {:heliotrope 1 :orange (hd/->seed)}))

(defn blerp-resonator-force
  [particle-id blerp-map]
  (hdd/clj->vsa* [:?= 1 [:. blerp-map particle-id]]))

;; ---------

(defn blerp-overlap
  [letter world blerp-activations]
  (def letter letter)
  (def world world)
  (def blerp-activations blerp-activations)
  (f/sum (f/bit-and (f/<= 1 blerp-activations)
                    (f/eq letter world))))


;; Most naive idea:
;; - overlaps for each letter in the alphabet
;;

(defn bottom-up-blerp-vote
  [field alphabet world]
  (into []
        (for [letter alphabet]
          (blerp-overlap letter
                         world
                         (pyutils/ensure-jvm
                          (p/read-activations field))))))


;; ---------

(defn update-glooby-1
  "Updates glooby based on blerp votes.

  Args:
    config: A map containing :blerp-map and :alphabet.
    blerps: A map of blerp identities to their corresponding fields.
    world: The current world state.

  Returns:
    A map of blerp identities to their chosen letters."
  [{:keys [blerp-map alphabet]} blerps world]
  (let [vote-matrices (for [[blerp-identity field] blerps]
                        [blerp-identity (bottom-up-blerp-vote field alphabet world)])

        ;; Create a map of blerp identities to their vote tensors
        blerp-votes (into {} vote-matrices)

        ;; Initialize sets for tracking
        available-letters (set alphabet)
        assigned-blerps #{}
        results {}]

    (loop [current-letters available-letters
           current-blerps (set (keys blerps))
           current-results results]
      (if (or (empty? current-letters) (empty? current-blerps))
        current-results
        (let [;; Find the highest vote across all remaining blerps and letters
              [best-blerp best-letter]
              (apply max-key
                     (fn [[blerp letter]]
                       (get-in blerp-votes [blerp letter] Double/NEGATIVE_INFINITY))
                     (for [blerp current-blerps
                           letter current-letters]
                       [blerp letter]))]
          (recur
           (disj current-letters best-letter)
           (disj current-blerps best-blerp)
           (assoc current-results best-blerp best-letter)))))))

(defn update-glooby
  [{:keys [blerp-map alphabet]} blerps world]
  (def alphabet alphabet)
  (def blerps blerps)
  {:alphabet alphabet
   :blerp-map
     (hdd/clj->vsa*
       (update-vals
         (update-glooby-1 {:alphabet alphabet} blerps world)
         (fn [v]
           (hd/drop-randomly (hdd/clj->vsa* 1) (- 1 v)))))}
  ;; (let [spec
  ;;       (for [[blerp-identity field] blerps]
  ;;         [blerp-identity
  ;;          (bottom-up-blerp-vote field alphabet
  ;;          world)])
  ;;       ;; ([:orange     [39.0 3.0]]
  ;;       ;;  [:heliotrope [41.0 4.0]])
  ;;       votes
  ;;       (apply
  ;;        map vector
  ;;        (map second spec))
  ;;       (dtt/reduce-axis votes dtype-argops/argmax)
  ;;       ]
  ;;   (reduce
  ;;    (fn [{:keys [blerps-left-over winners]} votes]
  ;;      )
  ;;    votes)
  ;;   (dtype-argops/argmax)
  ;;   (dtype-argops/argmax
  ;;    (dtt/select
  ;;     (first votes)
  ;;     (clojure.set/difference
  ;;      (into #{} alphabet)
  ;;      #{0})))
  ;;   ;; (map second spec)
  ;;   ;; (for [letter alphabet]
  ;;   ;;   ()
  ;;   ;;   )
  ;;   )
)



(comment
  (berp-resonator-force :heliotrope)
  (berp-resonator-force :heliotrope))
