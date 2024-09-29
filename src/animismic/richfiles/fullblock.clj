(defn full-block
  ([])
  ([n]
   (->
    (lib/->entity
     :rect
     {:color (defs/color-map (rand-nth
                              ;; [:cyan :white]
                              [:white :heliotrope]))
      :full-block? true
      :lifetime 0.1
      :transform
      (lib/->transform [(* n 200) 0] 200 (* 2 (q/height)) 1)})
    (lib/live
     (lib/every-n-seconds 0.1
                          (fn [e s k]
                            (assoc e :color (defs/color-map (rand-nth [:cyan :white])))))))))


(comment

  (swap! lib/event-queue
         (fnil conj [])
         (fn [s] (lib/append-ents s [(full-block 1)])))

  (def block-n (atom 0))
  (def dir (atom dec))

  (swap!
   lib/event-queue (fnil conj [])
   (fn [s]
     (lib/live s
               [:fo
                (lib/every-n-seconds
                 0.2
                 ;; (fn [] (lib/normal-distr 0.2 0.1))
                 (fn [s k]

                   (lib/append-ents
                    s
                    [(full-block (swap! block-n
                                        (comp #(mod % 15)
                                              @dir)))])))])))


  (swap!
   lib/event-queue (fnil conj [])
   (fn [s]
     (-> s
         (lib/live [:swap-dir


                    (lib/every-n-seconds
                     (fn [] (lib/normal-distr 5 2))
                     (fn [s k]

                       (swap! dir {dec inc inc dec})
                       s))])
         (lib/live
          [:fo
           (lib/every-n-seconds
            0.2
            ;; (fn [] (lib/normal-distr 0.2 0.1))
            (fn [s k]
              (lib/append-ents
               s
               [(full-block (swap! block-n
                                   (comp #(mod % 15)
                                         @dir)))])))]))))






  (let [block-n (atom 0)
        dir (atom dec)]
    (swap!
     lib/event-queue (fnil conj [])
     (fn [s]
       (-> s
           (lib/live [(random-uuid)
                      (lib/every-n-seconds
                       (fn [] (lib/normal-distr 5 2))
                       (fn [s k]

                         (swap! dir {dec inc inc dec})
                         s)
                       )])
           (lib/live
            [(random-uuid)
             (lib/every-n-seconds
              0.2
              ;; (fn [] (lib/normal-distr 0.2 0.1))
              (fn [s k]
                (lib/append-ents
                 s
                 [(full-block (swap! block-n
                                     (comp #(mod % 15)
                                           @dir)))])))]))))))
