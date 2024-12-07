(ns arc.arc-grid
  (:import [java.security MessageDigest])
  (:require [clojure.string :as str]
            [clojure.walk :as walk]
            [clojure.java.io :as io]
            [clojure.data.json :as json]))


;; (-> (bennischwerdtner.arc-editor.arc-editor/puzzle)
;;     :test
;;     :input)

(defn sha256 [string]
  (let [digest (.digest (MessageDigest/getInstance "SHA-256") (.getBytes string "UTF-8"))]
    (apply str (map (partial format "%02x") digest))))

(defn picture-hash [grid] (sha256 (prn-str grid)))

(defn ->picture [grid] {:grid grid :id (picture-hash grid)})

;; -------------------
;; puzzle:
;;
;; +----+-----+-----+-------------+
;; | pi | p,..| po  |             | <------ picture-trajectory
;; +----+-----+-----+             |
;; |                              |
;; |                              |
;; +------------------------------+
;;
;; picture:
;;
;;  is one 1 grid
;;  +--+--+
;;  |  |  |
;;  +--+--+
;;  |  | 1|
;;  +--+--+
;;
;;  max: 30x30
;;  min: 1x1
;;  colors: 0-9
;;
;;
;; picture-trajectory:
;;
;; [ input-picture, p2, p3, p4, ... , output-picture ]
;;
;;
;; puzzle:
;;
;; multiple picture-trajectories
;;
;; <- the conntent of 1 arc data json file
;;


;;
;; ---------------------
;; picture edge:
;;
;; - edge between 2 pictures
;; - (optionally) `transform(p1) -> p2`
;; - (optionally) inverse(transform)
;; ---------------------
;;
;;

;;
;; directed graph:
;;
;;
;;    edge
;;  a  ->  b
;;
;; edges:
;;
;; [a b] -> edge-data
;;
;;
;; picture graph:
;;


(defn ->edge [a b] [a b])

;; get in [ a b ] -> edge

;; get in graph a -> b


;; ---------------------------------------

(defn puzzle
  []
  (let
    [file
       ;; (io/file
       ;;  "/home/benj/repos/ARC-AGI/data/training/3631a71a.json")
       ;; (io/file
       ;;  "/home/benj/repos/ARC-AGI/data/training/3aa6fb7a.json")
       (first
         (shuffle
           (filter #(str/ends-with? % ".json")
             (file-seq
               (io/file
                 "/home/benj/repos/ARC-AGI/data/training/")))))
     data (json/read (io/reader file) :key-fn keyword)]
    (assoc data
      :file (str file)
      :arc-id (str/replace (java.io.File/.getName file)
                           #"\.json$"
                           ""))))


(defn rand-grid
  []
  (torch/tensor (-> (puzzle)
                    :test
                    first
                    :input)))


;; ---------------------------------------------------




(comment
  (puzzle))
